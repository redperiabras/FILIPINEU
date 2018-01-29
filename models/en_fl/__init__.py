import os
import pickle

from modules.model import NMT

if (os.path.isfile(os.getcwd()+'/models/en_fl/en-fl.nlm')):
	with open(os.getcwd()+'/models/en_fl/en-fl.nlm', 'rb') as f:
		config = pickle.load(f)
		model = NMT('en-fl', config)
		model.load(f)

		f.close()
else:
	model = None