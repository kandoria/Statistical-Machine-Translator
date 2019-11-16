from collections import defaultdict
import numpy as np
import utils

def get_logvectors(ls1,ls2):
    '''Returns tf vector representations of documents'''
    def get_vector(d):
        vec = np.array([v for v in d.values()], dtype=float)
        for i,v in enumerate(vec):
            if(v>0):
                vec[i] = 1+np.log10(v)
            else:
                vec[i] = 0
        
        return vec#/np.linalg.norm(vec)
        
    d1 = defaultdict(int)
    d2 = defaultdict(int)
    
    for s1 in ls1:
        s1 = utils.preprocess_string(s1).split()
        for w in s1:
            d1[w] += 1
            d2[w] = d2[w]
    for s2 in ls2:
        s2 = utils.preprocess_string(s2).split()
        for w in s2:
            d2[w] += 1
            d1[w] = d1[w]
    
    return get_vector(d1),get_vector(d2)

def cosine_similarity(ls1,ls2):
    '''Computes cosine similarity of two documents'''
    v1,v2 = get_logvectors(ls1,ls2)
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
#     print(v1.shape[0])
#     print(v2.shape[0])
    return np.dot(v1,v2)

def jaccard_coefficient(ls1,ls2):
    '''Computes jaccard coefficient of two documents'''
    set1 = set()
    set2 = set()
    for s1 in ls1:
        s1 = utils.preprocess_string(s1).split()
        for w in s1:
            set1.add(w)
    for s2 in ls2:
        s2 = utils.preprocess_string(s2).split()
        for w in s2:
            set2.add(w)
            
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / union
            