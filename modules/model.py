from bnas.model import Model, Linear, Embeddings, LSTMSequence
from bnas.optimize import Adam, iterate_batches
from bnas.init import Gaussian
from bnas.utils import softmax_3d
from bnas.loss import batch_sequence_crossentropy
from bnas.fun import function

from modules.sentence import tokenizer, detokenize
from modules.text import Encoded
from modules.search import beam_with_coverage

import sys
import theano
import numpy as np

from theano import tensor as T

class NMT(Model):
	def __init__(self, name, config):
		super().__init__(name)
		self.config = config

		self.add(Embeddings(
			'source_char_embeddings',
			1 if config['source_encoder'].sub_encoder is None \
					else len(config['source_encoder'].sub_encoder),
			config['source_char_embedding_dims'],
			dropout=config['char_embeddings_dropout']))

		self.add(Embeddings(
			'source_embeddings',
			len(config['source_encoder']),
			config['source_embedding_dims'],
			dropout=config['embeddings_dropout']))

		self.add(Embeddings(
			'target_embeddings',
			len(config['target_encoder']),
			config['target_embedding_dims']))

		self.add(Linear(
			'hidden',
			config['decoder_state_dims'],
			config['target_embedding_dims'],
			dropout=config['dropout'],
			layernorm=config['layernorm']))

		self.add(Linear(
			'emission',
			config['target_embedding_dims'],
			len(config['target_encoder']),
			w=self.target_embeddings._w.T))

		self.add(Linear(
			'proj_h0',
			config['encoder_state_dims'],
			config['decoder_state_dims'],
			dropout=config['dropout'],
			layernorm=config['layernorm']))

		self.add(Linear(
			'proj_c0',
			config['encoder_state_dims'],
			config['decoder_state_dims'],
			dropout=config['dropout'],
			layernorm=config['layernorm']))

		# The total loss is
		#   lambda_o*xent(target sentence) + lambda_a*xent(alignment)
		self.lambda_o = theano.shared(
				np.array(1.0, dtype=theano.config.floatX))
		self.lambda_a = theano.shared(
				np.array(config['alignment_loss'], dtype=theano.config.floatX))
		for prefix, backwards in (('fwd', False), ('back', True)):
			self.add(LSTMSequence(
				prefix+'_char_encoder', backwards,
				config['source_char_embedding_dims'] + (
					(config['source_embedding_dims'] // 2) if backwards else 0),
				config['source_embedding_dims'] // 2,
				layernorm=config['encoder_layernorm'],
				dropout=config['recurrent_dropout'],
				trainable_initial=True,
				offset=0))
		for prefix, backwards in (('fwd', False), ('back', True)):
			self.add(LSTMSequence(
				prefix+'_encoder', backwards,
				config['source_embedding_dims'] + (
					config['encoder_state_dims'] if backwards else 0),
				config['encoder_state_dims'],
				layernorm=config['encoder_layernorm'],
				dropout=config['recurrent_dropout'],
				trainable_initial=True,
				offset=0))
		self.add(LSTMSequence(
			'decoder', False,
			config['target_embedding_dims'],
			config['decoder_state_dims'],
			layernorm=config['decoder_layernorm'],
			dropout=config['recurrent_dropout'],
			attention_dims=config['attention_dims'],
			attended_dims=2*config['encoder_state_dims'],
			trainable_initial=False,
			contextgate=(config['decoder_gate'] == 'context'),
			offset=-1))

		h_t = T.matrix('h_t')
		self.predict_fun = function(
				[h_t],
				T.nnet.softmax(self.emission(T.tanh(self.hidden(h_t)))))

		inputs = T.lmatrix('inputs')
		inputs_mask = T.bmatrix('inputs_mask')
		chars = T.lmatrix('chars')
		chars_mask = T.bmatrix('chars_mask')
		outputs = T.lmatrix('outputs')
		outputs_mask = T.bmatrix('outputs_mask')
		attention = T.tensor3('attention')

		self.x = [inputs, inputs_mask, chars, chars_mask]
		self.y = [outputs, outputs_mask, attention]

		self.encode_fun = function(self.x,
			self.encode(*self.x))
		self.xent_fun = function(self.x+self.y,
			self.xent(*(self.x+self.y)))
		self.pred_fun = function(self.x+self.y[:-1],
			self(*(self.x+self.y[:-1])))

		# stats
		#self.beam_ends = np.zeros((config['max_target_length'],))

	def xent(self, inputs, inputs_mask, chars, chars_mask,
			 outputs, outputs_mask, attention):
		pred_outputs, pred_attention = self(
				inputs, inputs_mask, chars, chars_mask, outputs, outputs_mask)
		outputs_xent = batch_sequence_crossentropy(
				pred_outputs, outputs[1:], outputs_mask[1:])
		# Note that pred_attention will contain zero elements for masked-out
		# character positions, to avoid trouble with log() we add 1 for zero
		# element of attention (which after multiplication will be removed
		# anyway).
		batch_size = attention.shape[1].astype(theano.config.floatX)
		attention_mask = (inputs_mask.dimshuffle('x', 1, 0) *
						  outputs_mask[1:].dimshuffle(0, 1, 'x')
						  ).astype(theano.config.floatX)
		epsilon = 1e-6
		attention_xent = (
				   -attention[1:]
				 * T.log(epsilon + pred_attention + (1-attention_mask))
				 * attention_mask).sum() / batch_size
		return outputs_xent, attention_xent

	def loss(self, *args):
		outputs_xent, attention_xent = self.xent(*args)
		return super().loss() + self.lambda_o*outputs_xent \
				+ self.lambda_a*attention_xent

	def search(self, inputs, inputs_mask, chars, chars_mask,
			   max_length, beam_size=8, others=[], **kwargs):
		# list of models in the ensemble
		models = [self] + others
		n_models = len(models)
		n_states = 2

		# tuple (h_0, c_0, attended) for each model in the ensemble
		models_init = [m.encode_fun(inputs, inputs_mask, chars, chars_mask)
					   for m in models]

		# precomputed sequences for attention, one for each model
		models_attended_dot_u = [
				m.decoder.attention_u_fun()(model_init[-1])
				for m, model_init in zip(models, models_init)]

		# output embeddings for each model
		models_embeddings = [
				m.target_embeddings._w.get_value(borrow=False)
				for m in models]


		def step(i, states, outputs, outputs_mask, sent_indices):
			models_result = [
					models[idx].decoder.step_fun()(
						models_embeddings[idx][outputs[-1]],
						states[idx*n_states+0],
						states[idx*n_states+1],
						models_init[idx][-1][:,sent_indices,...],
						models_attended_dot_u[idx][:,sent_indices,...],
						inputs_mask[:,sent_indices])
					for idx in range(n_models)]
			mean_attention = np.array(
					[models_result[idx][-1] for idx in range(n_models)]
				 ).mean(axis=0)
			models_predict = np.array(
					[models[idx].predict_fun(models_result[idx][0])
					 for idx in range(n_models)])
			dist = models_predict.mean(axis=0)
			return ([x for result in models_result for x in result[:n_states]],
					dist, mean_attention)

		initial = [x for h_0, c_0, _ in models_init for x in [h_0, c_0]]
		result, i = beam_with_coverage(
				step,
				initial,
				models_init[0][0].shape[0],
				self.config['target_encoder']['<S>'],
				self.config['target_encoder']['</S>'],
				max_length,
				inputs_mask,
				alpha=self.config['alpha'], 
				beta=self.config['beta'],
				gamma=self.config['gamma'],
				len_smooth=self.config['len_smooth'],
				**kwargs)
		#self.beam_ends[i] += 1
		return result

	def encode(self, inputs, inputs_mask, chars, chars_mask):
		# First run a bidirectional LSTM encoder over the unknown word
		# character sequences.
		embedded_chars = self.source_char_embeddings(chars)
		fwd_char_h_seq, fwd_char_c_seq = self.fwd_char_encoder(
				embedded_chars, chars_mask)
		back_char_h_seq, back_char_c_seq = self.back_char_encoder(
				T.concatenate([embedded_chars, fwd_char_h_seq], axis=-1),
				chars_mask)

		# Concatenate the final states of the forward and backward character
		# encoders. These form a matrix of size:
		#   n_chars x source_embedding_dims
		# NOTE: the batch size here is n_chars, which is the total number of
		# unknown words in all the sentences in the inputs matrix.
		# Create an empty matrix if there are no unknown words
		# (e.g. pure word-level encoder)
		char_vectors = theano.ifelse.ifelse(T.gt(chars.shape[0], 0),
				T.concatenate([fwd_char_h_seq[-1], back_char_h_seq[0]], axis=-1),
				T.zeros([0, self.config['source_embedding_dims']],
				dtype=theano.config.floatX))

		# Compute separate masks for known words (with input symbol >= 0)
		# and unknown words (with input symbol < 0).
		known_mask = inputs_mask * T.ge(inputs, 0)
		unknown_mask = inputs_mask * T.lt(inputs, 0)
		# Split the inputs matrix into two, one indexing unknown words (from
		# the char_vectors matrix) and the other known words (from the source
		# word embeddings).
		unknown_indexes = (-inputs-1) * unknown_mask
		known_indexes = inputs * known_mask

		# Compute the final embedding sequence by mixing the known word
		# vectors with the character encoder output of the unknown words.
		# If there is no character encoder, just use the known word vectors.
		embedded_unknown = char_vectors[unknown_indexes]
		embedded_known = self.source_embeddings(known_indexes)
		embedded_inputs = theano.ifelse.ifelse(T.gt(chars.shape[0], 0),
				(unknown_mask.dimshuffle(0,1,'x').astype(
					theano.config.floatX) * embedded_unknown) + \
				(known_mask.dimshuffle(0,1,'x').astype(
					theano.config.floatX) * embedded_known),
				known_mask.dimshuffle(0,1,'x').astype(
					theano.config.floatX) * embedded_known)

		# Forward encoding pass
		fwd_h_seq, fwd_c_seq = self.fwd_encoder(embedded_inputs, inputs_mask)
		# Backward encoding pass, using hidden states from forward encoder
		back_h_seq, back_c_seq = self.back_encoder(
				T.concatenate([embedded_inputs, fwd_h_seq], axis=-1),
				inputs_mask)
		# Initial states for decoder
		h_0 = T.tanh(self.proj_h0(back_h_seq[0]))
		c_0 = T.tanh(self.proj_c0(back_c_seq[0]))
		# Attention on concatenated forward/backward sequences
		attended = T.concatenate([fwd_h_seq, back_h_seq], axis=-1)
		return h_0, c_0, attended

	def __call__(self, inputs, inputs_mask, chars, chars_mask,
				 outputs, outputs_mask):
		embedded_outputs = self.target_embeddings(outputs)
		h_0, c_0, attended = self.encode(
				inputs, inputs_mask, chars, chars_mask)
		h_seq, c_seq, attention_seq = self.decoder(
				embedded_outputs, outputs_mask, h_0=h_0, c_0=c_0,
				attended=attended, attention_mask=inputs_mask)
		pred_seq = softmax_3d(self.emission(T.tanh(self.hidden(h_seq))))

		return pred_seq, attention_seq

	def create_optimizer(self):
		return Adam(
				self.parameters(),
				self.loss(*(self.x + self.y)),
				self.x, self.y,
				grad_max_norm=5.0)

	def average_parameters(self, others):
		for name, p in self.parameters():
			p.set_value(np.mean(
					[p.get_value(borrow=True)] + \
					[other.parameter(name).get_value(borrow=True)
					 for other in others],
					axis=0))

	def prepare_batch(self, batch_pairs):
		src_batch, trg_batch, links_maps_batch = list(zip(*batch_pairs))
		x = self.config['source_encoder'].pad_sequences(src_batch, fake_hybrid=True)
		y = self.config['target_encoder'].pad_sequences(trg_batch)
		y = y + (np.ones(y[0].shape + (x[0].shape[0],),dtype=theano.config.floatX),)
		return x, y

	def translate(self, sents, encode=False, nbest=0, batch_size=32, beam_size=8,
		max_target_len=1000):
		for i in range(0, len(sents), batch_size):
			batch_sents = sents[i:i+batch_size]
			
			if encode:
				batch_sents = [self.config['source_encoder'].encode_sequence(sent)
							   for sent in batch_sents]

			x = self.config['source_encoder'].pad_sequences(
					batch_sents, fake_hybrid=True)
			
			beams = self.search(
					*(x + (max_target_len,)),
					beam_size=beam_size,
					prune=(nbest == 0))

			nbest = min(nbest, beam_size)

			for batch_sent_idx, (_, beam) in enumerate(beams):
				lines = []
				for best in list(beam)[:max(1, nbest)]:
					encoded = Encoded(best.history + (best.last_sym,), None)
					decoded = self.config['target_encoder'].decode_sentence(encoded)
					hypothesis = detokenize(
						decoded[::-1] if self.config['backwards'] else decoded,
						self.config['target_tokenizer'])
					if nbest > 0:
						lines.append(' ||| '.join((str(i+batch_sent_idx), hypothesis, str(best.norm_score))))
					else:
						yield hypothesis
				if lines:
					yield '\n'.join(lines)