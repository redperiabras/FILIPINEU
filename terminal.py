import os
import sys
import random
import pickle
import traceback

import numpy as np
import theano
from theano import tensor as T

from modules.sentence import tokenizer, read, detokenize
from modules.text import Encoded
from modules.model import NMT
from modules.bleu import BLEU
from modules.chrF import chrF
from modules.search import beam_with_coverage

from prompter import prompt, yesno
from collections import Counter
from arghandler import ArgumentHandler, subcmd
from datetime import datetime
from time import time
from pprint import pprint

from nltk import word_tokenize

from app import logger as log

@subcmd('initdb', help="Initialize Database")
def initdb(parser, context, args):

	from app import db

	db.create_all()
	log.info('SQL database created.', 'green')

@subcmd('dropdb', help='Drop Database')
def dropdb(parser, context, args):

	from app import db

	if yesno('Are you sure you want to lose all your sql data?', default='yes'):	
		db.drop_all()
		log.info('SQL database has been deleted.', 'green')

@subcmd('runserver', help="Run the Server")
def runserver(parser, context, args):

	from app import app

	app.run(
		host=app.config['HOST'],
		port=app.config['PORT'],
		debug=app.config['DEBUG'])

@subcmd('create-encoder', help="Encoder tool")
def create_encoder(parser, context, args):

	parser.add_argument('--files', required=True, metavar='FILE', type=str,
		nargs='+',
		help='File to process')

	parser.add_argument('--vocabulary-size', type=int, default=50000,
		help='Maximum number of word in the vocabulary')

	parser.add_argument('--char-size', type=int, default=200,
		help='Maximum number of characters in the vocabulary')

	parser.add_argument('--char-count', type=int, default=1,
		help='Minimum count of characters in the vocabulary (all characters are\
			included by default)')

	parser.add_argument('--tokenizer', type=str, default='word',
		choices=('space', 'char', 'word'),
		help='Tokenizer flag (space, char, word)')

	parser.add_argument('--lowercase', action='store_true',
		help='Lowercase the data')

	parser.add_argument('--hybrid', action='store_true',
		help='Create a hybrid word/character vocabulary')

	parser.add_argument('--save-to', required=True, metavar='FILE', type=str,
		help='Output file name')

	args = parser.parse_args(args)

	from modules.text import TextEncoder

	if args.tokenizer == 'char':
		tokenize = lambda s: list(s.strip())
	elif args.tokenizer == 'space' or args.tokenizer == 'bpe':
		tokenize = str.split
	elif args.tokenizer == 'word':
		import nltk
		from nltk import word_tokenize as tokenize

	token_count = Counter()
	char_count = Counter()

	character = args.tokenizer == 'char'    

	for filename in args.files:
		log.info('Processing %s' % os.path.basename(filename))
		with open(filename, 'rt') as f:
			for line in f:
				line = line.lower() if args.lowercase else line
				tokens = tokenize(line)
				token_count.update(tokens)
				if args.hybrid:
					char_count.update(''.join(tokens))
			f.close()

	log.info('Creating %s encoder' % os.path.splitext(args.save_to)[0])
	if args.hybrid:
		char_encoder = TextEncoder(
				counts = char_count,
				min_count = char_count,
				max_vocab = args.char_size,
				special=('<UNK>',))
		encoder = TextEncoder(
				counts = token_count,
				max_vocab = args.vocabulary_size,
				sub_encoder = char_encoder)
	else:
		encoder = TextEncoder(
				counts=token_count, 
				max_vocab=args.vocabulary_size,
				min_count=char_count if args.tokenizer == 'char' else None,
				special=('<S>', '</S>') + (() if args.tokenizer == 'char' else ('<UNK>',))
			)

	if os.path.isfile(os.path.splitext(args.save_to)[0] + ".vocab"):
		if not yesno('Encoder already exist. Replace?', default='yes'):
			args.save_to = prompt('%s' % log('Enter new encoder name:'))

	log.info('Exporting %s encoder' % os.path.basename(args.save_to))
	with open(os.path.splitext(args.save_to)[0] + ".vocab", 'wb') as f:
		pickle.dump(encoder, f, -1)
		f.close()

	log.info('Success')

@subcmd('create-model', help="Create new model")
def create_model(parser, context, args):

	parser.add_argument('--source-encoder', type=str, metavar='FILE',
		default=None, required=True,
		help='load source vocabulary ')

	parser.add_argument('--target-encoder', type=str, metavar='FILE',
		default=None, required=True,
		help='load target vocabulary')

	parser.add_argument('--source-tokenizer', type=str,
		choices=('word', 'space', 'char', 'bpe'), default='word',
		help='Type of Preprocessing source text')

	parser.add_argument('--target-tokenizer', type=str,
		choices=('word', 'space', 'char', 'bpe'), default='char',
		help='Type of Preprocessing target text')

	parser.add_argument('--alpha', type=float, default=0.01, metavar='X',
		help='Length penalty weight during beam translation')

	parser.add_argument('--beta', type=float, default=0.4, metavar='X',
		help='Coverage penalty weight during beam translation')

	parser.add_argument('--gamma', type=float, default=1.0, metavar='X',
		help='Over attention penalty weight during beam translation')

	parser.add_argument('--decoder-gate', type=str,
		choices=('lstm', 'context'), default='lstm',
		help='Tyoe of decoder gate (lstm or context)')

	parser.add_argument('--len-smooth', type=float, default=5.0, metavar='X',
		help='Smoothing constant for length penalty during beam translation')

	parser.add_argument('--word-embedding-dims', type=int, metavar='N',
		default=256, 
		help='Size of word embeddings')

	parser.add_argument('--target-embedding-dims', type=int, metavar='N',
		default=None, 
		help='Size of target embeddings (default: size of input word or char embedding')

	parser.add_argument('--char-embedding-dims', type=int, metavar='N',
		default=64, 
		help='Size of character embeddings')

	parser.add_argument('--dropout', type=float, metavar='FRACTION',
		default=0.0, 
		help='Use dropout for non-recurrent connections with the given factor')

	parser.add_argument('--encoder-state-dims', type=int, metavar='N',
		default=256, 
		help='Size of encoder state')

	parser.add_argument('--decoder-state-dims', type=int, metavar='N',
		default=512, 
		help='Size of decoder state')

	parser.add_argument('--attention-dims', type=int, metavar='N',
		default=256, 
		help='Size of attention vectors')

	parser.add_argument('--alignment-loss', type=float, metavar='X',
		default=0.0, 
		help='Alignment cross-entropy contribution to loss function (DEPRECATED)')

	parser.add_argument('--alignment-decay', type=float, metavar='X',
		default=0.9999, 
		help='Decay factor of alignment cross-entropy contribution (DEPRECATED)')

	parser.add_argument('--layer-normalization', action='store_true',
		help='Use layer normalization')

	parser.add_argument('--recurrent-dropout', type=float, metavar='FRACTION',
		default=0.0, 
		help='Use dropout for recurrent connections with the given factor')
	
	parser.add_argument('--source-lowercase', action='store_true',
		help='Convert source text to lowercase before processing')

	parser.add_argument('--target-lowercase', action='store_true',
		help='convert target text to lowercase before processing')

	parser.add_argument('--backwards', action='store_true',
		help='Reverse the order (token level) of all input data')

	parser.add_argument('--save-model', type=str, metavar='FILE',
		default=None, required=True,
		help='Output Model file')

	args = parser.parse_args(args)

	log.info('Loading Source language encoder')
	with open(args.source_encoder, 'rb') as f:
		args.source_encoder = pickle.load(f)
		f.close()

	log.info('Loading Target language encoder')
	with open(args.target_encoder, 'rb') as f:
		args.target_encoder = pickle.load(f)
		f.close()

	if args.target_embedding_dims is None:
		args.target_embedding_dims = (
			args.char_embedding_dims
			if args.target_tokenizer == 'char'
			else args.word_embedding_dims)

	log.info('Configuring model')
	config = {
		'ts_train': 0, 											# total training time in seconds
		'tn_epoch': 0, 											# total number of epochs
		'source_encoder': args.source_encoder,					#
		'target_encoder': args.target_encoder,					#
		'source_lowercase': args.source_lowercase,				# False
		'source_tokenizer': args.source_tokenizer,				# word
		'target_lowercase': args.target_lowercase,				# False
		'target_tokenizer': args.target_tokenizer,				# char
		'source_embedding_dims': args.word_embedding_dims,		# 256
		'source_char_embedding_dims': args.char_embedding_dims,	# 64
		'target_embedding_dims': args.target_embedding_dims,	# None
		'char_embeddings_dropout': args.dropout,				# 0.0
		'embeddings_dropout': args.dropout,						# 0.0
		'recurrent_dropout': args.recurrent_dropout,			# 0.0
		'dropout': args.dropout,								# 0.0
		'encoder_state_dims': args.encoder_state_dims,			# 256
		'decoder_state_dims': args.decoder_state_dims,			# 512
		'attention_dims': args.attention_dims,					# 256
		'layernorm': args.layer_normalization,					# False
		'alignment_loss': args.alignment_loss,					# 0.0
		'alignment_decay': args.alignment_decay,				# 0.9999
		'backwards': args.backwards,							# False
		'decoder_gate': args.decoder_gate,						# lstm
		'alpha': args.alpha,									# 0.01
		'beta': args.beta,										# 0.4
		'gamma': args.gamma,									# 1.0
		'decoder_gate': args.decoder_gate,						# lstm
		'len_smooth': args.len_smooth,							# 5.0
		'encoder_layernorm': 'ba2' if args.layer_normalization else False,
		'decoder_layernorm': 'ba2' if args.layer_normalization else False
	}

	if not config['source_encoder'].sub_encoder:
		log.warning('Source encoder is not hybrid')

	log.info('Checking existence')
	if os.path.isfile(args.save_model):
		if not yesno(log('Model %s exist, replace? ' % os.path.basename(args.save_model)), default='yes'):
			args.save_model = prompt(log('New model name: '))

	log.info('Creating model')
	model = NMT('nmt', config)
	
	log.info('Saving %s' % os.path.basename(args.save_model))
	with open(args.save_model, 'wb') as f:
		pickle.dump(config, f)
		model.save(f)
		f.close()

	log.info('Model Saved')

@subcmd('train', help="Model Trainer")
def train(parser, context, args):

	parser.add_argument('--load-model', type=str, metavar='FILE(s)',
		default=None, required=True,
		help='Existing Model file')

	parser.add_argument('--translate-every', type=int, default=250, metavar='N',
		help='Translate test set every N training batches')

	parser.add_argument('--train-data', type=str, metavar='FILE',
		required=True,
		help='Name of the Training data file')

	parser.add_argument('--source-test-data', type=str, metavar='FILE',
		default=None, required=True,
		help='Name of the source test-set file')

	parser.add_argument('--target-test-data', type=str, metavar='FILE',
		default=None, required=True,
		help='Name of the target test-set file')

	parser.add_argument('--batch-size', type=int, default=32, metavar='N',
		help='Minibatch size of training set')

	parser.add_argument('--batch-budget', type=float, default=32, metavar='N',
		help='Minibatch budget during training. The optimal value depends on model \
				size and available GPU memory. Try values between 20 and 200')

	parser.add_argument('--reset-optimizer', action='store_true',
		help='Reset model optimizer')

	parser.add_argument('--min-char-count', type=int, metavar='N',
		help='Drop all characters with count < N in training data')

	parser.add_argument('--learning-rate', type=float, default=None, metavar='X',
		help='Override the default learning rate for optimizer with X')

	parser.add_argument('--random-seed', type=int, default=123, metavar='N',
		help='Random seed for repeatable sorting of data')

	parser.add_argument('--max-target-length', type=int, default=1000, metavar='N',
		help='Maximum length of target sentence during translation (unit given by --target-tokenizer)')

	parser.add_argument('--reference', type=str, metavar='FILE', default=None,
		help='Name of the reference translation file')

	parser.add_argument('--nbest-list', type=int, default=0, metavar='N',
		help='Print n-best list in translation model')

	parser.add_argument('--beam-size', type=int, default=8, metavar='N',
		help='Beam size during translation')

	parser.add_argument('--train-for', type=int, default=1, metavar='N',
		help='Train for N epochs')

	args = parser.parse_args(args)

	from bnas.optimize import iterate_batches
	from modules.largetext import ShuffledText, HalfSortedIterator

	random.seed(args.random_seed)

	with open(args.load_model, 'rb') as f:
		log.info('Loading %s configuration' % os.path.basename(args.load_model))
		config = pickle.load(f)
		model = NMT('nmt', config)

		log.info('Loading %s weights' % os.path.basename(args.load_model))
		model.load(f)

		log.info('Initializing Optimizer')
		optimizer = model.create_optimizer()

		if args.learning_rate is not None:
			optimizer.learning_rate = args.learning_rate

		if not args.reset_optimizer:
			try:
				optimizer.load(f)
				log.info('Continuing traning from Epoch %d, Update %d' % (config['tn_epoch'], optimizer.n_updates))
			except EOFError:
				pass

	logf = open(args.load_model +
		'.log', 'a', encoding='utf-8')
	evalf = open(args.load_model +
		'-eval.log', 'a', encoding='utf-8')

	log.info('Initializing Source Text Tokenizer')
	source_tokenizer = tokenizer(config['source_tokenizer'],
		lowercase=config['source_lowercase'])

	log.info('Initializing Target Text Tokenizer')
	target_tokenizer = tokenizer(config['target_tokenizer'], 
		lowercase=config['target_lowercase'])

	log.info('Loading Source Language Testing data')
	source_test_data = read(args.source_test_data,
		source_tokenizer, config['backwards'])
	
	log.info('Loading Target Language Testing data')
	target_test_data = read(args.target_test_data,
		target_tokenizer, config['backwards'])

	target_test_unencoded = target_test_data
	source_test_data = [config['source_encoder'].encode_sequence(sent)
							for sent in source_test_data]
	target_test_data = [config['target_encoder'].encode_sequence(sent)
							for sent in target_test_data]
	test_link_maps = [(None, None, None)]*len(source_test_data)
	test_pairs = list(zip(source_test_data, target_test_data, test_link_maps))

	source_sample_data = source_test_data
	target_sample_data = target_test_unencoded

	log.info('Loading Training Data')
	training_data = open(args.train_data, 'rb')
	shuffled_training_data = ShuffledText(training_data, seed=args.random_seed)

	def validate(test_pairs, start_time, optimizer, logf, sent_nr):
		result = 0.
		att_result = 0.
		t0 = time()
		for batch_pairs in iterate_batches(test_pairs, args.batch_size):
			test_x, test_y = model.prepare_batch(batch_pairs)
			test_outputs, test_outputs_mask, test_attention = test_y
			test_xent, test_xent_attention = model.xent_fun(*(test_x + test_y))
			scale = (test_outputs.shape[1] / (test_outputs_mask.sum()*np.log(2)))
			result += test_xent * scale
			att_result += test_xent_attention*scale

		print('%d\t%.3f\t%.3f\t%.3f\t%d\t%d' %
			(
				int(t0 - start_time),		# Starting Time
				result,						# Result
				att_result,					# Attention Result
				time() - t0,				# End Time
				optimizer.n_updates,		# Number of Update
				sent_nr						# Number of Sentence Processed
				), file=logf, flush=True)

		return result

	epochs, batch_nr, sent_nr = 0, 0, 0
	chrf_max, bleu_max = 0.0, 0.0

	log.info('Starting training')
	log.info('Press Ctrl + C to stop training...')
	try:
		while epochs < args.train_for:

			start_time = time()

			train_samples = HalfSortedIterator(
						iter(shuffled_training_data),
						max_area=args.batch_budget*0x200,
						source_tokenizer=source_tokenizer,
						target_tokenizer=target_tokenizer,
						length=lambda pair: sum(map(len, pair)))

			for sent_pairs in train_samples:

				print('Number of Sentences: %d' % len(sent_pairs))

				source_batch, target_batch = list(zip(*sent_pairs))

				if config['backwards']:
					source_batch = [source_sent[::-1] for source_sent in source_batch]
					target_batch = [target_sent[::-1] for target_sent in target_batch]

				source_batch = [config['source_encoder'].encode_sequence(source_sent)
								for source_sent in source_batch]
				
				target_batch = [config['target_encoder'].encode_sequence(target_sent)
								for target_sent in target_batch]

				batch_link_maps = [(None, None, None)]*len(source_batch)

				batch_pairs = list(zip(source_batch, target_batch, batch_link_maps))

				sent_nr += len(batch_pairs)

				x, y = model.prepare_batch(batch_pairs)

				t0 = time()
				train_loss = optimizer.step(*(x + y))
				train_loss *= (y[0].shape[1] / (y[1].sum()*np.log(2)))
				log.info('Batch %2d:%4d has loss %.4f (%.2f s)' % (
					epochs + 1,
					optimizer.n_updates,
					train_loss,
					time() - t0))
				if np.isnan(train_loss):
					log.warning('NaN loss, aborting')
					sys.exit(1)

				model.lambda_a.set_value(np.array(
					model.lambda_a.get_value() * config['alignment_decay'],
					dtype=theano.config.floatX))

				batch_nr += 1

			epochs += 1
			config['tn_epoch'] += 1
			config['ts_train'] += time() - start_time

			#Validate Model
			validate(test_pairs, start_time, optimizer, logf, sent_nr)

			#Test Translate
			t0 = time()
			test_dec = list(model.translate(source_sample_data, encode=False))
			for source, target, test in zip(
				source_sample_data, target_sample_data, test_dec):
				log.info('Source:' )
				log.info('%s' % detokenize(
					config['source_encoder'].decode_sentence(source),
					config['source_tokenizer']))
				log.info('')
				log.info('Target:')
				log.info('%s' % detokenize(target, config['target_tokenizer']))
				log.info('')
				log.info('Output:')
				log.info('%s' % test)
				log.info('-'*40)
			log.info('Translation finished %.2f s' % (time() - t0))

			if config['target_tokenizer'] == 'char':
				system = [ detokenize(word_tokenize(s), 'space')
							for s in test_dec ]
				reference = [ detokenize(word_tokenize( detokenize(s, 'char')), 'space')
							for s in target_sample_data]
			else:
				system = test_dec
				reference = [ detokenize(s, config['target_tokenizer'])
								for s in target_sample_data ]

			bleu_result = BLEU(system, [reference])
			chrf_result = chrF(reference, system)
			is_best = chrf_result[0] >= chrf_max
			chrf_max = max(chrf_result[0], chrf_max)
			bleu_max = max(bleu_result[0], bleu_max)
			log.info('BLEU = %f (%f, %f, %f, %f, BP = %f)' % bleu_result)
			log.info('chrF = %f (precision = %f, recall = %f)' % chrf_result)

			if evalf is not None:
				print('%d\t%.3f\t%.3f\t%d\t%d' % (
					int(t0 - start_time),
					bleu_result[0],
					chrf_result[0],
					optimizer.n_updates,
					sent_nr
				), file=evalf, flush=True)

			if is_best:
				log.info('Marking as best model...')
				with open(args.load_model + ".best", 'wb') as f:
					pickle.dump(config, f)
					model.save(f)
					optimizer.save(f)
					f.close()

			#Save Model
			filename = os.path.dirname(args.load_model) + '/%s-%d-%d.nlm' % (
				os.path.splitext(
					os.path.basename(args.load_model))[0],
				config['tn_epoch'],
				optimizer.n_updates)
			
			log.info('Saving model at Epoch %d, Batch %d' % (config['tn_epoch'], optimizer.n_updates))
			with open(filename, 'wb') as f:
				pickle.dump(config, f)
				model.save(f)
				optimizer.save(f)
				f.close()

			with open(args.load_model, 'wb') as f:
				pickle.dump(config, f)
				model.save(f)
				optimizer.save(f)
				f.close()

		log.info('Training Finished')

	except KeyboardInterrupt:
		log.info('Trainer Stopped')

	except Exception:
		log.warning('Exception found, see console...')
		print(traceback.format_exc())
	
	if logf: logf.close()
	if evalf: evalf.close()

@subcmd('evaluate', help='Evaluation tool')
def translator(parser, context, args):

	parser.add_argument('--load-model', type=str, metavar='FILE(s)',
		help='Model file(s) to load from')

	parser.add_argument('--source-eval', type=str, metavar='FILE',
		required=True,
		help='Sentence to translate')

	parser.add_argument('--target-eval', type=str, metavar='FILE',
		required=True,
		help='Reference file')

	parser.add_argument('--nbest-list', type=int, metavar='N',
		default=0,
		help='print n-best list in translation model')

	parser.add_argument('--random-seed', type=int, default=123, metavar='N',
		help='Random seed for repeatable sorting of data')

	parser.add_argument('--beam-size', type=int, default=8, metavar='N',
		help='Beam size during translation')

	args = parser.parse_args(args)

	from nltk import word_tokenize
	from nltk.translate.bleu_score import (modified_precision, closest_ref_length,
		brevity_penalty, SmoothingFunction, sentence_bleu, corpus_bleu)

	from fractions import Fraction

	weights = (0.25, 0.25, 0.25, 0.25)

	nbest = args.nbest_list if args.nbest_list else 0
	hypotheses = []
	p_numerators = Counter()	# Key = ngram order, and value = no. of ngram matches.
	p_denominators = Counter()	# Key = ngram order, and value = no. of ngram in ref.
	hyp_lengths, ref_lengths = 0, 0

	log.info('Initializing Model')
	with open(args.load_model, 'rb') as f:
		log.info('Loading %s configuration' % os.path.basename(args.load_model))
		config = pickle.load(f)
		model = NMT('nmt', config)

		log.info('Loading %s weights' % os.path.basename(args.load_model))
		model.load(f)
		f.close()

	log.info('Loading Source Evaluation data')	
	source_tokenizer = tokenizer(config['source_tokenizer'],
		lowercase=config['source_lowercase'])
	source_eval = read(args.source_eval, source_tokenizer, config['backwards'])

	log.info('Loading Target Evaluation data')
	target_tokenizer = tokenizer('word', 
		lowercase=config['target_lowercase'])
	references = read(args.target_eval, target_tokenizer, False)

	log.info('Translating...')
	
	output_file = open(os.path.dirname(args.source_eval) + '/result.data.eval',
		'w', encoding='utf-8')

	for i, sent in enumerate(model.translate(source_eval, encode=True, nbest=nbest)):
	    print(sent, file=output_file, flush=True)
	    hypotheses.append(word_tokenize(sent))

	output_file.close()

	print("Expected: " + corpus_bleu)

	log.info('Generating Data for Evaluation')
	evaluation_file = open(os.path.dirname(args.source_eval) + '/scores.data.eval.csv',
		'w', encoding='utf-8')
	
	for reference, hypothesis in zip(references, hypotheses):
	    
	    hyp_len = len(hypothesis)
	    ref_len = closest_ref_length(reference, hyp_len)
	    
	    hyp_lengths += hyp_len
	    ref_lengths += ref_len
	    
	    set_data = '%d,%d' % (ref_len, hyp_len)
	    
	    for i, _ in enumerate(weights, start=1):
	        p_i = modified_precision([reference], hypothesis, i)
	        p_numerators[i] += p_i.numerator
	        p_denominators[i] += p_i.denominator
	        set_data += ',%d,%d' % (p_i.numerator, p_i.denominator)
	        
	    set_data += ',%f' % sentence_bleu([reference], hypothesis)

	    print(set_data, file=evaluation_file, flush=True)
	        
	evaluation_file.close()

	bp = brevity_penalty(ref_lengths, hyp_lengths)

	p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
			for i, _ in enumerate(weights, start=1)]

	smoothing_function = SmoothingFunction().method0

	p_n = smoothing_function(p_n, references=references, hypothesis=hypothesis,
	                             hyp_len=hyp_len, emulate_multibleu=False)

	s = (w * math.log(p_i) for i, (w, p_i) in enumerate(zip(weights, p_n)))

	print("Result: " + math.exp(math.fsum(s)))

	log.info('Process finished')

if __name__ == '__main__':
	with open('logo.txt', 'r') as f:
	    text = f.read()
	    for line in text.split('\n'):
	        print(line)
	        f.close()
	handler = ArgumentHandler(enable_autocompletion=True,
		description='FILIPINEU: Filipino - English Neural Machine Translation')
	handler.run()