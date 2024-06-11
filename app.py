import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from weather_clustering_app import handle_outliers, reduce_dimension, cluster_data, fillna_groupby_mean, fillna_groupby_mode

# Fungsi untuk memuat data
def load_data(file_path):
    return pd.read_csv(file_path)

# Memuat data
data = load_data('dataset.csv')

# Fill missing values
fillna_groupby_mean(data, ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg'])
fillna_groupby_mode(data, ['ddd_car'])
    
# Drop missing values
data.dropna(inplace=True)

# Encode categorical data
label_encoder = LabelEncoder()
data['ddd_car']= label_encoder.fit_transform(data['ddd_car'])

# UI Streamlit
st.title("Weather Data Clustering")
station_id = st.selectbox("Pilih Station ID", data['station_id'].unique())
outlier_handling = st.selectbox("Handle Outliers", ['No', 'Yes'])
reduction_method = st.selectbox("Pilih Metode Reduksi Dimensi", ['PCA', 't-SNE'])
n_cluster = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=10, value=4)
plotting_method = st.selectbox("Pilih Jenis Plotting", ['KMeans', 'DBSCAN', 'Spectral Clustering', 'Gaussian Mixture Model'])
# save_model_option = st.checkbox("Simpan model ke file")

# Handle Outliers
if outlier_handling == 'Yes':
    data = handle_outliers(data)

# Filter data berdasarkan station_id
data_filtered = data[data['station_id'] == station_id]
X = data_filtered.drop(columns=['date','station_id'])

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduksi dimensi
X_reduced = reduce_dimension(X_scaled, reduction_method)

# Clustering
model, labels = cluster_data(X_reduced, plotting_method, n_cluster)

# Compute silhouette score
silhouette_avg = silhouette_score(X_reduced, labels)
st.write(f'Silhouette Score: {silhouette_avg}')

# Plotting hasil clustering
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels, cmap='viridis', s=50)
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend1)
ax.set_title(f'{plotting_method} Clustering with {reduction_method}')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
st.pyplot(fig)

# Menyimpan model jika opsi dipilih
# if save_model_option:
#     filename = f'{plotting_method}_clust_model.pkl'
#     save_model(model, filename)
#     st.success(f'Model disimpan ke file: {filename}')