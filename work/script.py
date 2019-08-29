import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer

''' Utils '''
progress_freq = 10000
def print_progress(counter, total):
    percentage = str(round(counter / total, 4) * 100) + '%'
    print('Progress: processed {0} items out of {1} ... ({2})'.format(counter, total, percentage))

''' Paths '''
file_path = '../../data_fixed/'
pickle_path = '../../data_pickled/'

''' Read in data '''
file_location = file_path + 'page_aug_fixed.csv'
df = pd.read_csv(file_location, delimiter=',', encoding='iso-8859-1')

''' Data preparation '''
data = df[['sessionnumber', 'pagetitle', 'eventtimestamp']]
data = data.drop([8621281,8621282])
data.eventtimestamp = data.eventtimestamp.astype(int)
data = data.sort_values(by='eventtimestamp')

''' Create session_dict'''
session_dict = {}
counter = 0
skip_counter = 0
print('Creating session_dict...')
for _, row in data.iterrows():
    try:
        session_number = row.sessionnumber
        page_title = row.pagetitle
        if session_number in session_dict:
            session_dict[session_number] = session_dict[session_number] + page_title
        else:
            session_dict[session_number] = page_title
    except:
        skip_counter = skip_counter + 1
    if counter % progress_freq == 0:
        print_progress(counter, 14760627)
    counter = counter + 1
print('Created session_dict, skipped {0} items'.format(skip_counter))

''' Vectorize '''
vectorizer = HashingVectorizer(n_features=20, encoding='iso-8859-1')
counter = 0
skip_counter = 0
print('Vectorizing...')
for key, value in session_dict.items():
    try:
        session_dict[key] = vectorizer.fit_transform([value]).toarray().flatten().tolist()
    except:
        skip_counter = skip_counter + 1
    if counter % progress_freq == 0:
        print_progress(counter, 6000000)
    counter = counter + 1
print('Done vectorizing, skipped {0} items'.format(skip_counter))

''' Dump session_dict to pickle'''
with open(pickle_path + 'session_dict_aug.pickle', 'wb') as handle:
    pickle.dump(session_dict, handle)
