import os
import sys
import random
import pickle
import traceback

import numpy as np
import theano
from theano import tensor as T

from bnas.optimize import iterate_batches
from modules.sentence import tokenizer, read, detokenize
from modules.text import TextEncoder, Encoded
from modules.largetext import ShuffledText, HalfSortedIterator
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

	parser.add_argument('--files', required=True, metavar='FILE', type=str, nargs='+',
		help='File to process')

	parser.add_argument('--vocabulary-size', type=int, default=50000,
		help='Maximum number of word in the vocabulary')

	parser.add_argument('--char-size', type=int, default=200,
		help='Maximum number of characters in the vocabulary')

	parser.add_argument('--char-count', type=int, default=1,
		help='Minimum count of characters in the vocabulary (all characters are included by default)')

	parser.add_argument('--tokenizer', type=str, default='word', choices=('space', 'char', 'word'),
		help='Tokenizer flag (space, char, word)')

	parser.add_argument('--lowercase', action='store_true',
		help='Lowercase the data')

	parser.add_argument('--hybrid', action='store_true',
		help='Create a hybrid word/character vocabulary')

	parser.add_argument('--save-to', required=True, metavar='FILE', type=str,
		help='Output file name')

	args = parser.parse_args(args)

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
		'ts_train': 0, #total training time in seconds
		'tn_epoch': 0, #total number of epochs
		'source_encoder': args.source_encoder,
		'target_encoder': args.target_encoder,
		'source_lowercase': args.source_lowercase,
		'source_tokenizer': args.source_tokenizer,
		'target_lowercase': args.target_lowercase,
		'target_tokenizer': args.target_tokenizer,
		'source_embedding_dims': args.word_embedding_dims,
		'source_char_embedding_dims': args.char_embedding_dims,
		'target_embedding_dims': args.target_embedding_dims,
		'char_embeddings_dropout': args.dropout,
		'embeddings_dropout': args.dropout,
		'recurrent_dropout': args.recurrent_dropout,
		'dropout': args.dropout,
		'encoder_state_dims': args.encoder_state_dims,
		'decoder_state_dims': args.decoder_state_dims,
		'attention_dims': args.attention_dims,
		'layernorm': args.layer_normalization,
		'alignment_loss': args.alignment_loss,
		'alignment_decay': args.alignment_decay,
		'backwards': args.backwards,
		'decoder_gate': args.decoder_gate,
		'alpha': args.alpha,
		'beta': args.beta,
		'gamma': args.gamma,
		'decoder_gate': args.decoder_gate,
		'len_smooth': args.len_smooth,
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

	epochs = 0
	batch_nr = 0
	sent_nr = 0

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
	source_tokenizer = tokenizer(config['source_tokenizer'], lowercase=config['source_lowercase'])

	log.info('Initializing Target Text Tokenizer')
	target_tokenizer = tokenizer(config['target_tokenizer'], lowercase=config['target_lowercase'])

	log.info('Loading Source Language Testing data')
	source_test_data = read(args.source_test_data,
		source_tokenizer,
		config['backwards'])
	
	log.info('Loading Target Language Testing data')
	target_test_data = read(args.target_test_data,
		target_tokenizer,
		config['backwards'])

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

	def prepare_batch(batch_pairs):
		src_batch, trg_batch, links_maps_batch = list(zip(*batch_pairs))
		x = config['source_encoder'].pad_sequences(src_batch, fake_hybrid=True)
		y = config['target_encoder'].pad_sequences(trg_batch)
		y = y + (np.ones(y[0].shape + (x[0].shape[0],),dtype=theano.config.floatX),)
		return x, y

	def validate(test_pairs, start_time, optimizer, logf, sent_nr):
		result = 0.
		att_result = 0.
		t0 = time()
		for batch_pairs in iterate_batches(test_pairs, args.batch_size):
			test_x, test_y = prepare_batch(batch_pairs)
			test_outputs, test_outputs_mask, test_attention = test_y
			test_xent, test_xent_attention = model.xent_fun(*(test_x + test_y))
			scale = (test_outputs.shape[1] / (test_outputs_mask.sum()*np.log(2)))
			result += test_xent * scale
			att_result += test_xent_attention*scale

		print('%d\t%.3f\t%.3f\t%.3f\t%d\t%d' %
			(
				int(t0 - start_time),
				result,
				att_result,
				time() - t0,
				optimizer.n_updates,
				sent_nr
				), file=logf, flush=True)

		return result

	def translate(sents, encode=False, nbest=0, backwards=False):
		for i in range(0, len(sents), args.batch_size):
			batch_sents = sents[i:i+args.batch_size]
			
			if encode:
				batch_sents = [config['source_encoder'].encode_sequence(sent)
							   for sent in batch_sents]
			x = config['source_encoder'].pad_sequences(
					batch_sents, fake_hybrid=True)
			beams = model.search(
					*(x + (args.max_target_length,)),
					beam_size=args.beam_size,
					alpha=config['alpha'],
					beta=config['beta'],
					gamma=config['gamma'],
					len_smooth=config['len_smooth'],
					prune=(nbest == 0))
			nbest = min(nbest, args.beam_size)
			for batch_sent_idx, (_, beam) in enumerate(beams):
				lines = []
				for best in list(beam)[:max(1, nbest)]:
					encoded = Encoded(best.history + (best.last_sym,), None)
					decoded = config['target_encoder'].decode_sentence(encoded)
					hypothesis = detokenize(
						decoded[::-1] if backwards else decoded,
						config['target_tokenizer'])
					if nbest > 0:
						lines.append(' ||| '.join((str(i+batch_sent_idx), hypothesis, str(best.norm_score))))
					else:
						yield hypothesis
				if lines:
					yield '\n'.join(lines)

	chrf_max = 0.0
	bleu_max = 0.0

	start_time = time()
	total_train_time = 0

	log.info('Starting training')
	log.info('Press Ctrl + C to stop training...')
	try:
		while epochs < args.train_for:

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

				x, y = prepare_batch(batch_pairs)

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

			epochs += 1

			#Validate Model
			validate(test_pairs, start_time, optimizer, logf, sent_nr)

			#Test Translate
			t0 = time()
			test_dec = list(translate(source_sample_data, encode=False))
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
				system = [
					detokenize(word_tokenize(s), 'space')
						for s in test_dec]
				reference = [
					detokenize(word_tokenize(
						detokenize(s, 'char')), 'space')
							for s in target_sample_data]
			else:
				system = test_dec
				reference = [
					detokenize(s, config['target_tokenizer'])
						for s in target_sample_data]

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
				config['tn_epoch'] += epochs
				config['ts_train'] += time() - start_time
				with open(args.load_model + ".best", 'wb') as f:
					pickle.dump(config, f)
					model.save(f)
					optimizer.save(f)
					f.close()

			#Save Model
			filename = os.path.dirname(args.load_model) + '/%s-%d-%d.nlm' % (
				os.path.splitext(
					os.path.basename(args.load_model))[0],
				config['tn_epoch'] + epochs,
				optimizer.n_updates)

			config['tn_epoch'] += epochs
			config['ts_train'] += time() - start_time
			
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

	total_train_time = time() - start_time  
	config['tn_epoch'] += epochs
	config['ts_train'] += total_train_time

	log.info('Model trained in %6d seconds for %d batches' % (total_train_time, batch_nr))

@subcmd('translate', help='Translator tool')
def translator(parser, context, args):

	parser.add_argument('--load-model', type=str, metavar='FILE(s)',
		help='Model file(s) to load from')

	parser.add_argument('--translate', type=str, metavar='FILE',
		help='Sentence to translate')

	parser.add_argument('--reference', type=str, metavar='FILE',
		help='Reference file')

	parser.add_argument('--output', type=str, metavar='FILE',
		help='Output Transalations')

	parser.add_argument('--encode', action='store_false',
		help="Encode sentence")

	parser.add_argument('--nbest-list', type=int,
		default=0,
		metavar='N',
		help='print n-best list in translation model')

	parser.add_argument('--random-seed', type=int, default=123, metavar='N',
		help='Random seed for repeatable sorting of data')

	parser.add_argument('--beam-size', type=int, default=8, metavar='N',
		help='Beam size during translation')

	args = parser.parse_args(args)

	random.seed(args.random_seed)

	with open(args.load_model, 'rb') as f:
		log.info('Loading %s configuration' % os.path.basename(args.load_model))
		config = pickle.load(f)
		model = NMT('nmt', config)

		log.info('Loading %s weights' % os.path.basename(args.load_model))
		model.load(f)

		if args.learning_rate is not None:
			optimizer.learning_rate = args.learning_rate

		if not args.reset_optimizer:
			try:
				optimizer.load(f)
				log.info('Continuing traning from update %d' % optimizer.n_updates)
			except EOFError:
				pass

	def translate(sents, encode=False, nbest=0, backwards=False):
		for i in range(0, len(sents), config['batch_size']):
			batch_sents = sents[i:i+config['batch_size']]
			if encode:
				batch_sents = [config['src_encoder'].encode_sequence(sent)
							   for sent in batch_sents]
			x = config['src_encoder'].pad_sequences(
					batch_sents, fake_hybrid=True)
			beams = model.search(
					*(x + (args.max_target_length,)),
					beam_size=config['beam_size'],
					alpha=config['alpha'],
					beta=config['beta'],
					gamma=config['gamma'],
					len_smooth=config['len_smooth'],
					others=models[1:],
					prune=(nbest == 0))
			nbest = min(nbest, config['beam_size'])
			for batch_sent_idx, (_, beam) in enumerate(beams):
				lines = []
				for best in list(beam)[:max(1, nbest)]:
					encoded = Encoded(best.history + (best.last_sym,), None)
					decoded = config['trg_encoder'].decode_sentence(encoded)
					hypothesis = detokenize(
						decoded[::-1] if backwards else decoded,
						config['target_tokenizer'])
					if nbest > 0:
						lines.append(' ||| '.join((str(i+batch_sent_idx), hypothesis, str(best.norm_score))))
					else:
						yield hypothesis
				if lines:
					yield '\n'.join(lines)

	log.info('Transalating...')
	outf = sys.stdout if args.output is None else open(args.output, 'w', encoding='utf-8')
	sents = read_sents(args.translate,tokenize_src,config['backwards'] == 'yes')

	if args.reference: hypotheses = []
	if args.nbest_list: nbest = args.nbest_list
	else: nbest = 0
	for i,sent in enumerate(translate(sents, encode=True, nbest=nbest, backwards=(config['backwards'] == 'yes'))):
		print('.', file=sys.stderr, flush=True, end='')
		print(sent, file=outf, flush=True)
		if args.reference:
			if nbest:
				hypotheses.append(sent.split('\n')[0].split(' ||| ')[1])
			else:
				hypotheses.append(sent)
	log.info(' done!')

	if args.output:
		outf.close()

	# compute BLEU if reference file is given
	if args.reference:
		# Now the translation is flipped, so the reference should not be
		# flipped
		trg = read_sents(args.reference,
				tokenize_trg, False)
		#config['backwards'] == 'yes')

		if config['target_tokenizer'] == 'char':
			system = [detokenize(word_tokenize(s),'space')
						for s in hypotheses]
			reference = [detokenize(word_tokenize(detokenize(s,'char')), 'space')
							for s in trg]
			print('BLEU = %f (%f, %f, %f, %f, BP = %f)' % BLEU(
				system,[reference]))
			print('chrF = %f (precision = %f, recall = %f)' % chrF(
				reference,system))
		else:
			reference = [detokenize(s,config['target_tokenizer'])
							for s in trg ]
			print('BLEU = %f (%f, %f, %f, %f, BP = %f)' % BLEU(
				hypotheses,[reference]))
			print('chrF = %f (precision = %f, recall = %f)' % chrF(
				reference,hypotheses))

if __name__ == '__main__':
	with open('logo.txt', 'r') as f:
	    text = f.read()
	    for line in text.split('\n'):
	        print(line)
	        f.close()
	handler = ArgumentHandler(enable_autocompletion=True,
		description='FILIPINEU: Filipino - English Neural Machine Translation')
	handler.run()