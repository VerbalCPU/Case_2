import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns; sns.set()  # for plot styling

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

plot = sys.argv[1] if len(sys.argv) > 1 else 'False'

''' Read in pickled data '''
pickle_path = '../../data_pickled/'
with open(pickle_path + 'session_dict_aug.pickle', 'rb') as handle:
    session_dict = pickle.load(handle)

data_vectorized = pd.DataFrame.from_dict(session_dict, orient='index')
data_vectorized = data_vectorized.sample(n=1000, random_state=1)
data_vectorized = data_vectorized.values

''' PCA '''
if plot == 'PCA':
    pca = PCA(n_components=6).fit(data_vectorized)

    pca_df = pd.DataFrame(
            {
                'Variance Explained':pca.explained_variance_ratio_,
                'Principal Components':['PC1','PC2','PC3','PC4','PC5','PC6']
            })
    ax = sns.barplot(x="Principal Components", y="Variance Explained", data=pca_df, color="c");
    labels = np.around(np.arange(0, 0.4, 0.1), 4)
    plt.yticks(labels, labels)
    plt.title('PCA')
    plt.show()

optimal_number_of_principal_components = 3
data_vectorized = PCA(n_components=optimal_number_of_principal_components).fit_transform(data_vectorized)

''' Feature scaling '''
sc = MinMaxScaler(feature_range=(0,1))
data_vectorized = sc.fit_transform(data_vectorized)

''' Training the SOM '''
import pdb; pdb.set_trace()
som = MiniSom(2, 2, data_vectorized.shape[1])
som.random_weights_init(data_vectorized)
som.train_random(data_vectorized, 100)


''' Vizualize '''
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
for i, x in enumerate(data_vectorized):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         's',
         markeredgecolor = 'g',
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
