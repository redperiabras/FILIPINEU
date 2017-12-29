import os
import pickle

from modules.model import NMT

with open(os.getcwd()+'/models/fl_en/fl-en.nlm.best', 'rb') as f:
	config = pickle.load(f)
	model = NMT('fl-en', config)
	model.load(f)

	f.close()
