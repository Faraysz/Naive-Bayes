import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from collections import Counter

# --- LOAD DATA IRIS DARI SKLEARN ---
iris = load_iris()

# Ambil hanya Petal Length (kolom ke-3) dan Petal Width (kolom ke-4)
Xdata = iris.data[:, 2:4]

# Target asli (0=Setosa, 1=Versicolor, 2=Virginica) -> samakan dengan MATLAB (1,2,3)
target = iris.target + 1  

# --- KMEANS CLUSTERING ---
k = 3
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
idx = kmeans.fit_predict(Xdata)

# --- MAP CLUSTER KE LABEL TARGET ---
kluster = np.zeros(k)
hasil = np.zeros(len(idx))

for j in range(1, k+1):
    a = np.where(target == j)[0]
    modus = Counter(idx[a]).most_common(1)[0][0]
    kluster[j-1] = modus

for i in range(len(idx)):
    if idx[i] == kluster[0]:
        hasil[i] = 1
    elif idx[i] == kluster[1]:
        hasil[i] = 2
    elif idx[i] == kluster[2]:
        hasil[i] = 3

# --- VISUALISASI ---
plt.figure()
plt.scatter(Xdata[idx == 0, 0], Xdata[idx == 0, 1], c='red', label='Cluster 1')
plt.scatter(Xdata[idx == 1, 0], Xdata[idx == 1, 1], c='green', label='Cluster 2')
plt.scatter(Xdata[idx == 2, 0], Xdata[idx == 2, 1], c='blue', label='Cluster 3')

# Centroid
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='x', s=200, c='black', label='Centroids')

plt.title('Clustering Iris Data Set dengan K-means')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.grid(True)
plt.show()

# --- HITUNG AKURASI ---
jumlah_data = len(hasil)
jumlah_benar = np.sum(hasil == target)
akurasi = jumlah_benar / jumlah_data * 100

print("Akurasi K-means: {:.2f}%".format(akurasi))
