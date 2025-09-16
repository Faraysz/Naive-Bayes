# Import libraries yang dibutuhkan
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Dataset: Cuaca, Temperatur, Angin, Olahraga
# Cuaca: Cerah=0, Hujan=1
# Temperatur: Normal=0, Tinggi=1
# Angin: Pelan=0, Kencang=1
# Olahraga: Tidak=0, Ya=1
 # Dataset: Cuaca, Temperatur, Angin, Olahraga
data = [
    ['Cerah', 'Normal', 'Pelan', 'Ya'],
    ['Cerah', 'Normal', 'Pelan', 'Ya'],
    ['Hujan', 'Tinggi', 'Pelan', 'Tidak'],
    ['Cerah', 'Normal', 'Kencang', 'Ya'],
    ['Hujan', 'Tinggi', 'Kencang', 'Tidak'],
    ['Cerah', 'Normal', 'Pelan', 'Ya']
]

# Memisahkan fitur (X) dan label (y)
X = [d[:3] for d in data]
y = [d[3] for d in data]

# Label encoding untuk mengubah fitur kategorikal menjadi numerik
# (cuaca, temperatur, angin)
le_cuaca = LabelEncoder()
le_temperatur = LabelEncoder()
le_angin = LabelEncoder()

X = np.array(X)
X[:, 0] = le_cuaca.fit_transform(X[:, 0])       # Cuaca
X[:, 1] = le_temperatur.fit_transform(X[:, 1])  # Temperatur
X[:, 2] = le_angin.fit_transform(X[:, 2])       # Angin

# Label encoding untuk hasil Olahraga (Ya/Tidak)
le_olahraga = LabelEncoder()
y = le_olahraga.fit_transform(y)

# Mengonversi fitur menjadi tipe data integer
X = X.astype(int)

# Inisialisasi model Naive Bayes
model = CategoricalNB()

# Melatih model
model.fit(X, y)

# Contoh prediksi: Cuaca=Cerah, Temperatur=Normal, Angin=Kencang
sample = np.array([[le_cuaca.transform(['Cerah'])[0],
                    le_temperatur.transform(['Normal'])[0],
                    le_angin.transform(['Kencang'])[0]]])

prediksi = model.predict(sample)
print("Apakah berolahraga?", le_olahraga.inverse_transform(prediksi)[0])

# Install scikit-learn
# !pip install scikit-learn
