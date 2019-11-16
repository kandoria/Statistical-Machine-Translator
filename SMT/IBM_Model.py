from collections import defaultdict
import math
import utils

# IBM Model 1
class IBM1:
    def __init__(self,t=None,wmap=None):
        """Initializing term probabilities and defining word dictionary and word mappings"""
        if(t==None):
            self.t = defaultdict(float)
            self.t.default_factory = (lambda:1.0)
        else:
            self.t = t
        if(wmap==None):
            self.wordmap = {}
        else:
            self.wordmap = wmap
        
    def run_iter(self, lang1_data, lang2_data, NumIter, lang1_vdata=None, lang2_vdata=None):
        """Auxilliary function to run EM iterations of the data passed"""
        bitext = []
        start = 0
        step = 10000
        while(start<len(lang1_data)):
            if(len(lang1_data)-start<step):
                bitext.append((lang1_data[start:], lang2_data[start:]))
            else:
                bitext.append((lang1_data[start:start+step], lang2_data[start:start+step]))
            start += step
            
        for k in range(NumIter):
            self.em_iter(bitext,k+1)
        self.update_maxprob_words()
        return
    
    def em_iter(self, sliced_bitext, iterNum):
        """Runs a single iteration of the EM algorithm on the model"""
        
        likelihood = 0.0
        c1 = defaultdict(float)
        c2 = defaultdict(float)
        for sentNum,(lang1_slice,lang2_slice) in enumerate(sliced_bitext):
        
            # Final Processing of data 
            lang1_slice = [sent.split() for sent in lang1_slice]
            lang2_slice = [sent.split() for sent in lang2_slice]
            corpus_slice = zip(lang1_slice, lang2_slice)

            # The E-Step
            for k,(e, d) in enumerate(corpus_slice):
                d = [None] + d
                l = len(d)    # l+1
                m = len(e)    # m
                q = 1.0       # alignment probability is equal for all unlike IBM 2

                for i in range(0, m):

                    num = [ q * self.t[(e[i], d[j])] for j in range(0,l) ]
                    den = float(sum(num))
                    likelihood += math.log(den)

                    for j in range(0, l):
                        delta = num[j] / den
                        c1[(e[i], d[j])] += delta
                        c2[(d[j],)]      += delta
                if(k%1000==999):
                    print('\rIter:%d Sentences:%d  '%(iterNum, sentNum*10000+k+1), end='')

        # The M-Step
        self.t = defaultdict(float, { k: (v/c2[k[1:]]) for k, v in c1.items() if v>0.0})
        print('\tLog-likelihood: %.5f'%(likelihood))

        return
    
    def update_maxprob_words(self):
        """Derives and stores the mapping of words to be used for translation"""
        worddict = defaultdict(defaultdict)
        for w1,w2 in self.t:
            worddict[w2][w1] = self.t[(w1,w2)]
        self.wordmap = {k:max(worddict[k], key=worddict[k].get) for k in worddict}
        return
    
    def get_translation(self, tsent):
        """Gets translation of a single sentence"""
        def translate_string(s):
            if(s.isnumeric()):
                return s
            else:
                try:
                    ret = self.wordmap[s]
                except:
                    ret = ''
                return ret
            
        if(type(tsent)==str):
            tsent = [translate_string(word) for word in utils.preprocess_string(tsent).split() if len(word)>0]
            return ' '.join(tsent)
        elif(type(tsent)==list):
            return [translate_string(word) for word in tsent]
        
    def translate(self, lang2_data):
        """Runs a translation of the data passed"""
        return [self.get_translation(tsent) for tsent in lang2_data]
    
# IBM Model 2
class IBM2:
    def __init__(self,t=None,q=None,wmap=None):
        """Initializing term probabilities, alignment probabilites and defining word dictionary and word mappings"""
        if(t==None):
            self.t = defaultdict(float)
            self.t.default_factory = (lambda:1.0)
        else:
            self.t = t
        if(q==None):
            self.q = defaultdict(float)
            self.q.default_factory = (lambda:1.0)
        else:
            self.q = q
        if(wmap==None):
            self.wordmap = {}
        else:
            self.wordmap = wmap
        
    def run_iter(self, lang1_data, lang2_data, NumIter, lang1_vdata=None, lang2_vdata=None):
        """Auxilliary function to run EM iterations of the data passed"""
        bitext = []
        start = 0
        step = 10000
        while(start<len(lang1_data)):
            if(len(lang1_data)-start<step):
                bitext.append((lang1_data[start:], lang2_data[start:]))
            else:
                bitext.append((lang1_data[start:start+step], lang2_data[start:start+step]))
            start += step
        
        for k in range(NumIter):
            self.em_iter(bitext,k+1)
        self.update_maxprob_words()
        return
    
    def em_iter(self, sliced_bitext, iterNum):
        """Runs a single iteration of the EM algorithm on the model"""
        
        likelihood = 0.0
        c1 = defaultdict(float)
        c2 = defaultdict(float)
        c3 = defaultdict(float)
        c4 = defaultdict(float)
        for sentNum,(lang1_slice,lang2_slice) in enumerate(sliced_bitext):
        
            # Final Processing of data 
            lang1_slice = [sent.split() for sent in lang1_slice]
            lang2_slice = [sent.split() for sent in lang2_slice]
            corpus_slice = zip(lang1_slice, lang2_slice)

            # The E-Step
            for k,(e, d) in enumerate(corpus_slice):
                d = [None] + d
                l = len(d)    # l+1
                m = len(e)    # m

                for i in range(0, m):

                    num = [ 10000 * self.q[(j, i, l, m)] * self.t[(e[i], d[j])] for j in range(0,l) ]
                    den = float(sum(num))
                    likelihood += math.log(den)

                    for j in range(0, l):
                        delta = num[j] / den
                        c1[(e[i], d[j])] += delta
                        c2[(d[j],)]      += delta
                        c3[(j, i, l, m)] += delta
                        c4[(i, l, m)]    += delta
                if(k%1000==999):
                    print('\rIter:%d Sentences:%d  '%(iterNum, sentNum*10000+k+1), end='')

        # The M-Step
        self.t = defaultdict(float, { k: (v/c2[k[1:]]) for k, v in c1.items() if v>0.0})
        self.q = defaultdict(float, { k: (v/c4[k[1:]]) for k, v in c3.items() if v>0.0})
        print('\tLog-likelihood: %.5f'%(likelihood))

        return
        
    def update_maxprob_words(self):
        """Derives and stores the mapping of words to be used for translation"""
        worddict = defaultdict(defaultdict)
        for w1,w2 in self.t:
            worddict[w2][w1] = self.t[(w1,w2)]
        self.wordmap = {k:max(worddict[k], key=worddict[k].get) for k in worddict}
        return
    
    def get_translation(self, tsent):
        """Gets translation of a single sentence"""
        def translate_string(s):
            if(s.isnumeric()):
                return s
            else:
                try:
                    ret = self.wordmap[s]
                except:
                    ret = ''
                return ret
            
        if(type(tsent)==str):
            tsent = [translate_string(word) for word in utils.preprocess_string(tsent).split() if len(word)>0]
            return ' '.join(tsent)
        elif(type(tsent)==list):
            return [translate_string(word) for word in tsent]
        
    def translate(self, lang2_data):
        """Runs a translation of the data passed"""
        return [self.get_translation(tsent) for tsent in lang2_data]