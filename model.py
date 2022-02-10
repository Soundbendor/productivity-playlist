import tensorflow as tf
import pandas as pd
import neptune
import pathlib
import numpy as np
import pprint
import helper

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def load_dataset(frames_path, val_split = 0):
    AUTOTUNE    = tf.data.experimental.AUTOTUNE
    points_pd  = pd.read_csv("{}/data.csv".format(frames_path), header=0, usecols=[0, 3, 4], index_col = 0)
    scaler = StandardScaler()
    points_pd[['arousal', 'valence']] = scaler.fit_transform(points_pd[['arousal', 'valence']])
    points = [(points_pd.iloc[i]['valence'], points_pd.iloc[i]['arousal']) for i in range(len(points_pd))]

    datasets = []

    if val_split != 0:

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            frames_path, labels=points, class_names=None, seed=123, validation_split=0.2, subset="training", color_mode='rgb', batch_size=batch_size, image_size=(img_height, img_width) 
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            frames_path, labels=points, class_names=None, seed=123, validation_split=0.2, subset="validation", color_mode='rgb', batch_size=batch_size, image_size=(img_height, img_width) 
        )

        train_ds = train_ds.cache().shuffle(buffer_size=10).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().shuffle(buffer_size=10).prefetch(buffer_size=AUTOTUNE)
        datasets = [train_ds, val_ds]
    
    else:
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            frames_path, labels=points, class_names=None, seed=123,
            color_mode='rgb', batch_size=batch_size, image_size=(img_height, img_width) 
        )

        ds = ds.cache().shuffle(buffer_size=10).prefetch(buffer_size=AUTOTUNE)
        datasets = [ds]

    print("=========== Dataset read ==============")
    print("Directory: {}".format(frames_path))
    for idx in range(len(datasets)):
        print("Dataset {} Length:".format(idx), tf.data.experimental.cardinality(datasets[idx]))
        for elem in datasets[idx].take(1): print(elem)
    print("=======================================")

    return datasets

batch_size  = 32
img_height  = 360
img_width   = 640
epochs      = 200
dropout     = 0.2
val_frac    = 0.2

PARAMS = {
    "batch_size"   : batch_size,
    "epochs"       : epochs,
    "dropout"      : dropout,
    "split_frac"   : val_frac
}

train_dsarr = load_dataset("./data/afftrain", val_frac)
test_dsarr  = load_dataset("./data/afftest")
train_ds = train_dsarr[0]
val_ds = train_dsarr[1]
test_ds = test_dsarr[0]

api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNThkNWU4ZDQtZWU0Mi00YmQ3LTk2MWMtMTEyNTQ0N2MwOWNiIn0="
class NeptuneMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # send logs
        for key in logs.keys():
            neptune.send_metric(key, epoch, logs[key])

        # TODO: run things on test set
        if epoch % 25 == 0:
            print(epoch)
            

        return

# neptune.init("Soundbendor/playlist", api_token=api_token)
# exp = neptune.create_experiment(params=PARAMS, upload_source_files=["model.py"])
# neptune_callback = NeptuneMonitor()

# data_augmentation = keras.Sequential([
#     layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
#     layers.experimental.preprocessing.RandomRotation(0.1),
# ])

# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
#     model = tf.keras.Sequential([
#         # data_augmentation,
#         layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#         layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
#         layers.MaxPooling2D(),
#         layers.Conv2D(32, 3, padding='same', activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(64, 3, padding='same', activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Dropout(dropout),  
#         layers.Flatten(),
#         layers.Dense(128, activation='sigmoid'),
#         layers.Dense(2, activation='tanh')
#     ])

#     model.compile(
#     optimizer='RMSprop',
#     loss='cosine_similarity',
#     metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_similarity'])

#     history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=epochs,
#     callbacks=[neptune_callback]
#     )

#     model.summary()
