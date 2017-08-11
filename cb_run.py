'''
Created on Aug 4, 2017

@author: Jacek Rozwadowski
'''

from optparse import OptionParser
import logging
from engine import ContentBaseEngine

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--redis_url", type="string", default="redis://localhost:6379",
              dest="s_redis_url", help="Redis database url.")

op.add_option("--data_url", type="string", default="example-data.csv",
              dest="s_data_url", help="Training data url.")

op.add_option("--training", action="store_true", default=False,
              help="Training mode")

op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")

op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")

op.add_option("--n-similar", type=int, default=10,
              help="Maximum number of similar items")

op.add_option("--item_id", type="string",
              dest="s_item_id", help="Item Id.")

op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")

#print(__doc__)
#op.print_help()
(opts, args) = op.parse_args()

print("")
print("Redis database: %s" % opts.s_redis_url)
print("Training Mode: %s" % opts.training)

if opts.training:
    print("Loading data from file: %s" % opts.s_data_url)
else:
    print("Similar items: %s" % opts.n_similar)
    print("Item id: %s" % opts.s_item_id)

print("")   

cb_engine = ContentBaseEngine(opts.s_redis_url) 

# Training
if opts.training:
    print("Training")
    cb_engine.train(opts.s_data_url, opts.use_hashing, opts.use_idf, opts.n_components)

# Prediction
else:
    print("Training")
    cb_engine.predict(opts.s_data_url, opts.s_item_id, opts.n_similar)
    
