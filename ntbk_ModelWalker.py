
# coding: utf-8

# In[1]:


import os
import pickle
from nltk import word_tokenize
from modules.model import NMT
from modules.sentence import tokenizer, read, detokenize


# In[ ]:


MODEL_FILES = 'models/en_fl'
#pabago nalang ng 'models/en_fl' to 'models/fl_en' pagkatapos tapos parun ulit. salamat :)

SAVEPOINTS = MODEL_FILES + '/model_savepoints'
SRC_EVAL = MODEL_FILES + '/source.data.eval'
TRG_EVAL = MODEL_FILES + '/target.data.eval'

bleu_results = []
model_list = os.listdir(SAVEPOINTS)

for nlm in model_list:
    print('Loading ' + nlm)
    with open(SAVEPOINTS + '/' + nlm, 'rb') as f:
        config = pickle.load(f)
        model = NMT('fl-en', config)
        model.load(f)
    f.close()
    
    print('Reloading source data...')
    source_tokenizer = tokenizer(config['source_tokenizer'], 
        lowercase=config['source_lowercase'])
    source = read(SRC_EVAL, source_tokenizer, config['backwards'])
    
    print('Reloading target data...')
    references = read(TRG_EVAL, word_tokenize, False)
    
    print('Translating...')
    hypotheses = []
    for i, sent in enumerate(model.translate(source, encode=True, nbest=0)):
        hypotheses.append(word_tokenize(sent))
        print(i)
        
    print('Starting Evaluation')
    bleu = corpus_bleu(reference, hypotheses)
    bleu_results.append(bleu)
    print('{} has BLEU score of {}'.format(nlm, bleu))
    
print('Best Model identified is' + model_list[bleu_results.index(max(bleu_results))])