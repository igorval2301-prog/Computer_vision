import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Завантаження даних
df = pd.read_csv("data/figures.csv")
print(df.head())

encoder = LabelEncoder()
df["label_enc"] = encoder.fit_transform(df["label"])

X = df[["area", "perimeter", "corners"]]
y = df["label_enc"]

# 2. Модель (8 нейронів на виході — це ок, якщо класів <= 8)
model = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=(3,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(8, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 3. Навчання
history = model.fit(X, y, epochs=200, verbose=0)

# 4. Візуалізація
plt.plot(history.history['loss'], label="ВТРАТИ")
plt.plot(history.history['accuracy'], label="ТОЧНІСТЬ") # Виправлено 'accurasy' -> 'accuracy'
plt.xlabel("ЕПОХА")
plt.ylabel("ЗНАЧЕННЯ") # Виправлено другий xlabel на ylabel
plt.title("ПРОЦЕС НАВЧАННЯ МОДЕЛІ")
plt.legend()
plt.show()

test = np.array([[16, 16, 0]])

pred = model.predict(test)

print(f"ЙМОВІРНІСТЬ КОЖНОГО КЛАСУ: {pred}")

print(f"РЕЗУЛЬТАТ: {encoder.inverse_transform([np.argmax(pred)])[0]}")