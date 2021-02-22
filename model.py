import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import pathlib
import os
import pprint

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

frames_path = "./data/affsample"
points_path = "./data/affsample/data.csv"

frames_dir = pathlib.Path(frames_path)
points_pd  = pd.read_csv(points_path, header=0, usecols=[0, 3, 4], index_col = 0)

batch_size = 32
img_height = 360
img_width  = 640
img_count  = len(points_pd)

points = [(points_pd.iloc[i]['valence'], points_pd.iloc[i]['arousal']) for i in range(img_count)]

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    frames_path, labels=points, class_names=None, seed=123, validation_split=0.2, subset="training", color_mode='rgb', batch_size=batch_size, image_size=(img_height, img_width) 
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    frames_path, labels=points, class_names=None, seed=123, validation_split=0.2, subset="validation", color_mode='rgb', batch_size=batch_size, image_size=(img_height, img_width) 
)

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     print(labels[i])
#     plt.axis("off")

# pprint.pprint(train_ds)
# pprint.pprint(val_ds)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(2)
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'])

model.fit(
  val_ds, 
#   validation_data=val_ds,
  epochs=1
)

model.summary()