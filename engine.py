'''
Created on Aug 4, 2017

Inspired by
http://scikit-learn.org/stable/modules/feature_extraction.html

@author: Jacek Rozwadowski
'''

import pandas as pd
import redis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import make_pipeline
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from time import time

class ContentBaseEngine(object):
    
    RKEY = 'p:smlr:%s'
    
    def __init__(self, redis_url):
        self._r = redis.StrictRedis.from_url(redis_url)
        
        
    def train(self, data_source, use_hashing=False, use_idf=True, desired_dimensionality=None):
        t0 = time()
        self._r.flushdb()
        print("0. Flush the stale training data from redis done in %fs" % (time() - t0))
        
        
        t0 = time()
        ds = pd.read_csv(data_source, sep=',', error_bad_lines=False, encoding='iso8859_2', low_memory=True)
        print("1. Read data from file done in %fs" % (time() - t0))
        
        
        if use_hashing:
            if use_idf:
                t0 = time()
                hasher = HashingVectorizer(analyzer='word', ngram_range=(1, 3), 
                                           stop_words='english', non_negative=True, 
                                           norm=None, binary=False)
                vectorizer = make_pipeline(hasher, TfidfTransformer())
                print("2. Perform an IDF normalization on the output of HashingVectorizer done in %fs" % (time() - t0))
            else:
                t0 = time()
                vectorizer = HashingVectorizer(analyzer='word', ngram_range=(1, 3), 
                                               stop_words='english', non_negative=False, norm='l2', binary=False)
                print("2. Perform pure HashingVectorizer done in %fs" % (time() - t0))
        else:
            t0 = time()
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), 
                                         min_df=0, stop_words='english')
            print("2. Perform TfidfVectorizer done in %fs" % (time() - t0))
            
            
        t0 = time()
        tfidf_matrix = vectorizer.fit_transform(ds['description'].values.astype('U'))
        print("3. Create a matrix of unigrams, bigrams, and trigrams done in %fs" % (time() - t0))
        
        
        if desired_dimensionality:
            t0 = time()
            svd = TruncatedSVD(desired_dimensionality)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)
            tfidf_matrix = lsa.fit_transform(tfidf_matrix)
            tfidf_matrix=sparse.csr_matrix(tfidf_matrix)
            print("4. Performing dimensionality reduction using LSA done in %fs" % (time() - t0))
            
            explained_variance = svd.explained_variance_ratio_.sum()
            print("    Explained variance of the SVD: {}%".format(int(explained_variance * 100)))
            
            
        t0 = time()
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
        print("5. Calculate cosine similarity done in %fs" % (time() - t0))
        
        
        t0 = time()
        for idx, row in ds.iterrows():
                similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
                similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]
    
                # First item is the item itself, so remove it.
                # This 'sum' is turns a list of tuples into a single tuple: [(1,2), (3,4)] -> (1,2,3,4)
                flattened = sum(similar_items[1:], ())
                self._r.zadd(self.RKEY % row['id'], *flattened)
        print("6. Save data in redis done in %fs" % (time() - t0))
        
        
    def predict(self, data_source, item_id=None, n_similar=10):
        print("1. Read data from file")
        ds = pd.read_csv(data_source, sep=',', error_bad_lines=False, encoding='iso8859_2', low_memory=True)
        ds[['id']] = ds[['id']].astype(str)
        
        
        t0 = time()
        ret_data = self._r.zrange(self.RKEY % item_id, 0, n_similar-1, withscores=True, desc=True)
        print("2. Read data from redis done in %fs" % (time() - t0))
        
        
        data = []
        for x in ret_data:
            data.append((x[0].decode(), x[1]))
        
        print("3. Similar items in descending order")   
        
        base_row = ds.loc[ds['id'] == item_id]
        desc = base_row['description'].to_string(index=False)
        print("   %s::%s" % (item_id, desc))  
           
        for r in data:
            sim_row = ds.loc[ds['id'] == r[0]]
            skey = sim_row['id'].to_string(index=False)
            sdesc = sim_row['description'].to_string(index=False)
            print("      %s::%s" % (skey, sdesc))
            