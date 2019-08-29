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

''' K Means '''
''' Elbow method for optimal k'''
if plot == 'Elbow':
    sum_of_squared_distances = []
    K = range(2,11)
    for k in K:
        print("Trying for {0} clusters...".format(k))
        km = KMeans(n_clusters=k, random_state=1)
        km = km.fit(data_vectorized)
        sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('K')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal K')
    plt.show()

''' Silhouette method for optimal k'''
if plot == 'Silhouette':
    silhouette = []
    K = range(2,11)
    for n_clusters in K:
        print("Trying for {0} clusters...".format(n_clusters))
        # Create a subplot with 1 row and 1 columns
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(data_vectorized) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 1 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=1)
        cluster_labels = clusterer.fit_predict(data_vectorized)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data_vectorized, cluster_labels)
        print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data_vectorized, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette  methotd with K = %d" % n_clusters),
                    fontsize=14, fontweight='bold')

    plt.show()

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k)
kmeans.fit(data_vectorized)
y_kmeans = kmeans.predict(data_vectorized)

plt.scatter(data_vectorized[:, 0], data_vectorized[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.show()
