import os
import pickle

from modules.model import NMT

if (os.path.isfile(os.getcwd()+'/models/fl_en/fl-en.nlm.best')):
	with open(os.getcwd()+'/models/fl_en/fl-en.nlm.best', 'rb') as f:
		config = pickle.load(f)
		model = NMT('fl-en', config)
		model.load(f)

		f.close()
else:
	model = None
