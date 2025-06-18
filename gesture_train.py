import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# === CONFIG ===
data_dir = "gesture_dataset"
img_size = 64

X = []
y = []
labels = os.listdir(data_dir)
label_dict = {label: idx for idx, label in enumerate(labels)}

# === LOAD & PREPROCESS DATA ===
for label in labels:
    folder_path = os.path.join(data_dir, label)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(img)
        y.append(label_dict[label])

X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
y = to_categorical(y)

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === BUILD CNN MODEL ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === TRAIN ===
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# === SAVE MODEL ===
model.save("gesture_model.h5")
print("\n✅ CNN model saved as gesture_model.h5")
