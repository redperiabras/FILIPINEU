import gzip

from collections import Counter
from nltk.tokenize import word_tokenize as tokenize

def tokenizer(mode, lowercase=False):
	if mode == 'char':
		if lowercase:
			tokenizer = (lambda s: list(s.strip().lower()))
		else:
			tokenizer = (lambda s: list(s.strip()))
	elif (mode == 'space') or (mode == 'bpe'):
		if lowercase:
			tokenizer = (lambda s: s.lower().split())
		else:
			tokenizer = str.split
	elif mode == 'word':
		if lowercase:
			tokenizer = (lambda s: tokenize(s.lower()))
		else:
			tokenizer = (lambda s: tokenize(s))
	else:
		raise ValueError('Unknown tokenizer: "%s"' % mode)
	
	return tokenizer

def detokenize(sent, name):
	#TODO use NLTK detokenizer
	if name == 'bpe':
		string = ' '.join(sent)
		return string.replace("@@ ", "")
	return ('' if name == 'char' else ' ').join(sent)

def read(filename, tokenizer, backwards, nbest=False):
	def process(line):
		if nbest:
			nr, text, score = line.split(' ||| ')
			line = text
		tokens = tokenizer(line)

		return tokens[::-1] if backwards else tokens

	if filename.endswith('.gz'):
		def open_func(fname):
			return gzip.open(fname, 'rt', encoding='utf-8')
	else:
		def open_func(fname):
			return open(fname, 'r', encoding='utf-8')

	with open_func(filename) as f:
		return list(map(process, f))