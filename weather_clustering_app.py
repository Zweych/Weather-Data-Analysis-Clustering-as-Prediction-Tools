import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl

# Fungsi untuk handle missing values
def fillna_groupby_mean(data, columns):
    for column in columns:
        data[column] = data.groupby('station_id')[column].transform(lambda x: x.fillna(x.mean()))

def fillna_groupby_mode(data, columns):
    for column in columns:
        data[column] = data.groupby('station_id')[column].transform(lambda x: x.fillna(x.mode()))

# Fungsi untuk mereduksi dimensi
def reduce_dimension(X, method, n_components=3):
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 'LDA':
        reducer = LDA(n_components=n_components)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Metode reduksi dimensi tidak dikenal")
    
    return reducer.fit_transform(X)

# Fungsi untuk clustering
def cluster_data(X, method):
    if method == 'KMeans':
        clusterer = KMeans(n_clusters=4, random_state=42)
    elif method == 'DBSCAN':
        clusterer = DBSCAN(eps=0.5, min_samples=2)
    elif method == 'Spectral Clustering':
        clusterer = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', random_state=42)
    elif method == 'Gaussian Mixture Model':
        clusterer = GaussianMixture(n_components=4, random_state=42)
    else:
        raise ValueError("Metode clustering tidak dikenal")
    
    labels = clusterer.fit_predict(X)
    return clusterer, labels

# Fungsi untuk plotting 3D
def plot_3d(X, labels, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=50)
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.show()

# Fungsi untuk menyimpan model
# def save_model(model, filename):
#     with open(filename, 'wb') as file:
#         pkl.dump(model, file)