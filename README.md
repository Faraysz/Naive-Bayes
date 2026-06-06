# 🌸 Naive Bayes & KNN Classification

Proyek pembelajaran mesin (Machine Learning) yang mengimplementasikan algoritma klasifikasi **Naive Bayes** dan **K-Nearest Neighbors (KNN)** menggunakan bahasa pemrograman Python.

Proyek ini bertujuan untuk memahami cara kerja algoritma klasifikasi dalam menangani dataset numerik (Iris) dan dataset kategorikal (Cuaca/Olahraga).

---

## ✨ Fitur
- ✅ Implementasi algoritma **K-Nearest Neighbors (KNN)** dari library `scikit-learn`
- ✅ Implementasi algoritma **Categorical Naive Bayes** dari `scikit-learn`
- ✅ Implementasi **Naive Bayes manual** (dari nol)
- ✅ Preprocessing data menggunakan `LabelEncoder` untuk data kategorikal
- ✅ Pembagian data training dan testing otomatis
- ✅ Evaluasi akurasi model

---

## 📁 Struktur Proyek
📊 Dataset
1. Dataset Bunga Iris (iris.csv)
Dataset klasik dari R.A. Fisher yang berisi 150 sampel bunga Iris dari 3 spesies:

    Iris Setosa
    Iris Versicolor
    Iris Virginica

Setiap sampel memiliki 4 fitur:

    Sepal Length (panjang sepal)
    Sepal Width (lebar sepal)
    Petal Length (panjang petal)
    Petal Width (lebar petal)

2. Dataset Cuaca/Olahraga
  Dataset kategorikal manual yang digunakan untuk memprediksi keputusan berolahraga berdasarkan:

    Cuaca: Cerah, Berawan, Hujan
    Temperatur: Panas, Sedang, Dingin
    Angin: Kencang, Lemah
    Keputusan: Ya / Tidak

⚙️ Persyaratan
Pastikan Anda telah menginstall:

    Python 3.7+
    Library berikut:


📈 Hasil
Setelah menjalankan program, Anda akan melihat:

    Prediksi kelas untuk setiap data testing
    Akurasi model dalam melakukan klasifikasi
    Perbandingan hasil prediksi dengan nilai sebenarnya (ground truth)

🧠 Algoritma yang Digunakan
Naive Bayes
Algoritma klasifikasi probabilistik berdasarkan Teorema Bayes dengan asumsi independen antar fitur. Sangat cocok untuk data kategorikal.
K-Nearest Neighbors (KNN)
Algoritma yang mengklasifikasikan data berdasarkan kedekatan jarak dengan K tetangga terdekat. Dalam proyek ini digunakan nilai k=3.

📄 Lisensi
Proyek ini dibuat untuk tujuan pembelajaran.

👤 Penulis
Faraysz
GitHub Profile
<div align="center">
  <i>Jika proyek ini bermanfaat, jangan lupa untuk memberikan ⭐ Star!</i>
</div>
```
