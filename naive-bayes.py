import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# === Baca Data dari CSV ===
df = pd.read_csv("iris.csv")

# Pisahkan data training dan data testing
train_data = df[df["Species"] != "?"]
test_data = df[df["Species"] == "?"]

# X = fitur, y = label
X_train = train_data[["SL","SW","PL","PW"]].values
y_train = train_data["Species"].values

X_test = test_data[["SL","SW","PL","PW"]].values

# === Model KNN ===
knn = KNeighborsClassifier(n_neighbors=3)  # k=3
knn.fit(X_train, y_train)

# Prediksi data uji
y_pred = knn.predict(X_test)

print("Data uji:", X_test)
print("Prediksi species:", y_pred[0])
