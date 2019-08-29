import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer

''' Read in data '''
file_path = '../../data_fixed/page_aug_fixed.csv'
df = pd.read_csv(file_path, delimiter=',', encoding='iso-8859-1', nrows=1000)

''' Data preparation '''
data = df[['sessionnumber', 'pagetitle', 'eventtimestamp']]
#data = data.drop([8621281,8621282])
data.eventtimestamp = data.eventtimestamp.astype(int)
data = data.sort_values(by='eventtimestamp')

session_dict = {}
counter = 0
import pdb; pdb.set_trace()
for _, row in data.iterrows():
    session_number = row.sessionnumber
    page_title = row.pagetitle
    if session_number in session_dict:
        session_dict[session_number] = session_dict[session_number] + page_title
    else:
        session_dict[session_number] = page_title

    if counter % 1000 == 0:
        print('Progress: processed {0} session numbers'.format(counter))
    counter = counter + 1

# Vectorize
vectorizer = HashingVectorizer(n_features=8, encoding='iso-8859-1')
import pdb; pdb.set_trace()
for key, value in session_dict.items():
    session_dict[key] = vectorizer.fit_transform([value]).toarray().flatten().tolist()

data_vectorized = pd.DataFrame.from_dict(session_dict, orient='index').values

import pdb; pdb.set_trace()
# PCA
pca = PCA(n_components=2)
data_vectorized = pca.fit_transform(data_vectorized)

# KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(data_vectorized)
y_kmeans = kmeans.predict(data_vectorized)

plt.scatter(data_vectorized[:, 0], data_vectorized[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.show()
