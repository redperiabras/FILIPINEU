import os
import pickle

from modules.model import NMT

if (os.path.isfile(os.getcwd()+'/models/en_fl/en-fl.nlm.best')):
	with open(os.getcwd()+'/models/en_fl/en-fl.nlm.best', 'rb') as f:
		config = pickle.load(f)
		model = NMT('fl-en', config)
		model.load(f)

		f.close()
else:
	model = None