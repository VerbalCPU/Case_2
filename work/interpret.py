import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns; sns.set()  # for plot styling

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from collections import Counter
from pprint import pprint

''' Utils '''
progress_freq = 10000
def print_progress(counter, total):
    percentage = str(round(counter / total, 4) * 100) + '%'
    print('Progress: processed {0} items out of {1} ... ({2})'.format(counter, total, percentage))

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

''' Read in pickled data '''
pickle_path = '../../data_pickled/'
with open(pickle_path + 'session_dict_aug_basic.pickle', 'rb') as handle:
    basic_dict = pickle.load(handle)

with open(pickle_path + 'session_dict_aug.pickle', 'rb') as handle:
    session_dict = pickle.load(handle)

with open(pickle_path + 'clus.pickle', 'rb') as handle:
    clusters = pickle.load(handle)

# Delete bad values
del basic_dict[88445588]
del basic_dict[88710979]
del basic_dict[90847293]
del basic_dict[91418116]
del basic_dict[94851312]

for i in clusters.keys():
    print("Most common session for cluster {0}".format(i))
    session_numbers = clusters[i]
    specific_basic_dict = {k: '<>'.join(basic_dict[k]) for k in session_numbers if all(isinstance(x, str) for x in basic_dict[k])}
    pprint(Counter(specific_basic_dict.values()).most_common()[0:10])
    print("====================================================")
