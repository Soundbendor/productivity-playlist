import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import pathlib
import os
import pprint
import neptune

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNThkNWU4ZDQtZWU0Mi00YmQ3LTk2MWMtMTEyNTQ0N2MwOWNiIn0="
frames_path = "./data/affsample"
points_pd  = pd.read_csv("{}/data.csv".format(frames_path), header=0, usecols=[0, 3, 4], index_col = 0)

batch_size = 32
img_height = 360
img_width  = 640
img_count  = len(points_pd)
epochs     = 10

PARAMS = {
    "dataset_name" : "affsample",
    "img_count"    : img_count,
    "batch_size"   : 32,
    "epochs"       : epochs,
    "dropout"      : 0.2
}

class NeptuneMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        neptune.send_metric("loss", epoch, logs["loss"])
        neptune.send_metric("val_loss", epoch, logs["val_loss"])
        neptune.send_metric("acc", epoch, logs["accuracy"])
        neptune.send_metric("val_acc", epoch, logs["val_accuracy"])

neptune.init("Soundbendor/playlist", api_token=api_token)
exp = neptune.create_experiment(params=PARAMS, upload_source_files=["model.py"])
neptune_callback = NeptuneMonitor()

points = [(points_pd.iloc[i]['valence'], points_pd.iloc[i]['arousal']) for i in range(img_count)]

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    frames_path, labels=points, class_names=None, seed=123, validation_split=0.2, subset="training", color_mode='rgb', batch_size=batch_size, image_size=(img_height, img_width) 
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    frames_path, labels=points, class_names=None, seed=123, validation_split=0.2, subset="validation", color_mode='rgb', batch_size=batch_size, image_size=(img_height, img_width) 
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])

model = tf.keras.Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2)
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'])

history = model.fit(
  train_ds, 
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[neptune_callback]
)

model.summary()