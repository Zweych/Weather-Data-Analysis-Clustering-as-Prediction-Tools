import numpy as np
from sklearn.decomposition import PCA
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
        
# Fungsi untuk handle outliers
def handle_outliers(data):
    for i in data.columns:
        if i not in ['station_id', 'date']:
            Q1 = data[i].quantile(0.25)
            Q3 = data[i].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[i] = np.where(data[i] < lower_bound, lower_bound, data[i])
            data[i] = np.where(data[i] > upper_bound, upper_bound, data[i])
    return data

# Fungsi untuk mereduksi dimensi
def reduce_dimension(X, method, n_components=3):
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    # elif method == 'LDA':
    #     reducer = LDA(n_components=n_components)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Metode reduksi dimensi tidak dikenal")
    
    return reducer.fit_transform(X)

# Fungsi untuk clustering
def cluster_data(X, method, n_cluster):
    if method == 'KMeans':
        clusterer = KMeans(n_clusters=n_cluster, random_state=42)
    elif method == 'DBSCAN':
        clusterer = DBSCAN(eps=0.5, min_samples=2)
    elif method == 'Spectral Clustering':
        clusterer = SpectralClustering(n_clusters=n_cluster, affinity='nearest_neighbors', random_state=42)
    elif method == 'Gaussian Mixture Model':
        clusterer = GaussianMixture(n_components=n_cluster, random_state=42)
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
    
# Function to assign descriptive labels to clusters
def assign_descriptive_labels(data, labels):
    unique_labels = set(labels)
    descriptive_labels = {}
    for label in unique_labels:
        cluster_data = data[labels == label]
        avg_RR = cluster_data['RR'].mean()
        avg_ss = cluster_data['ss'].mean()
        
        if avg_RR > 10 and avg_ss < 5:  # High rainfall, low sunshine
            descriptive_labels[label] = 'Hujan'  # Rain
        elif avg_RR < 5 and avg_ss > 5:  # Low rainfall, high sunshine
            descriptive_labels[label] = 'Cerah'  # Sunny
        elif 5 <= avg_RR <= 10 and avg_ss < 5:  # Medium rainfall, low to medium sunshine
            descriptive_labels[label] = 'Gerimis'  # Drizzle
        elif avg_RR < 10 and 5 <= avg_ss <= 10:  # Low to medium rainfall, medium sunshine
            descriptive_labels[label] = 'Berawan'  # Cloudy
        else:
            descriptive_labels[label] = 'Undefined'  # If none of the conditions match
    return descriptive_labels

# Fungsi untuk menyimpan model
# def save_model(model, filename):
#     with open(filename, 'wb') as file:
#         pkl.dump(model, file)