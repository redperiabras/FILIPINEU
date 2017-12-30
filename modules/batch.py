import theano
import numpy as np

def prepare_batch(batch_pairs):
	src_batch, trg_batch, links_maps_batch = list(zip(*batch_pairs))
	x = config['source_encoder'].pad_sequences(src_batch, fake_hybrid=True)
	y = config['target_encoder'].pad_sequences(trg_batch)
	y = y + (np.ones(y[0].shape + (x[0].shape[0],),dtype=theano.config.floatX),)
	return x, y