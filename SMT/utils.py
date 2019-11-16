import string
import IBM_Model
import pickle

def get_modelED():
    print("Loading Model...")
    with open('e2d_wordmap.pkl','rb') as f:
        model = IBM_Model.IBM2(wmap=pickle.load(f))
    print("Model loaded!")
    return model

def get_modelDE():
    print("Loading Model...")
    with open('d2e_wordmap.pkl','rb') as f:
        model = IBM_Model.IBM2(wmap=pickle.load(f))
    print("Model loaded!")
    return model

def preprocess_string(s):
    '''Returns processed string with digits'''
    return s.strip().translate(str.maketrans("","", string.punctuation)).lower()

def process_string(s):
    '''Returns processed string without digits used for translation'''
    return preprocess_string(s).translate(str.maketrans("","", string.digits))

def get_data(dataDir):
    '''Return a list of sentences(strings) from a file'''
    print("Reading Data...")
    try:
        with open(dataDir,'r') as f:
            languageFile = [ process_string(ln) for ln in f ]
    except:
        with open(dataDir,'r',encoding='utf-8') as f:
            languageFile = [ process_string(ln) for ln in f ]
    print("Data reading completed.")
    return languageFile

def run(ibm_model, lang1_data, lang2_data, NumIter, split=1.0):
    '''Useful for training the model while using a validation split'''
    train_size = int(split*len(lang1_data))
    ibm_model.run_iter(lang1_data[:train_size], lang2_data[:train_size], NumIter, lang1_data[train_size:], lang2_data[train_size:])
    return