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

frames_path = "./data/afftrain"
api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNThkNWU4ZDQtZWU0Mi00YmQ3LTk2MWMtMTEyNTQ0N2MwOWNiIn0="
image_dir   = pathlib.Path(frames_path)

points_pd  = pd.read_csv("{}/data.csv".format(frames_path), header=0, usecols=[0, 3, 4], index_col = 0)
scaler = MinMaxScaler(feature_range=(-1,1))
grid = np.transpose(np.array([points_pd.iloc[:,0], points_pd.iloc[:,1]]))
labels = scaler.fit_transform(grid)

batch_size  = 32
img_height  = 360
img_width   = 640
img_count   = len(points_pd)
epochs      = 200
dropout     = 0.2
val_frac    = 5

PARAMS = {
    "dataset_name" : frames_path[7:],
    "img_count"    : img_count,
    "batch_size"   : batch_size,
    "epochs"       : epochs,
    "dropout"      : dropout,
    "split_frac"   : val_frac
}

class NeptuneMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        neptune.send_metric("loss", epoch, logs["loss"])
        neptune.send_metric("val_loss", epoch, logs["val_loss"])
        
        neptune.send_metric("mean_squared_error", epoch, logs["mean_squared_error"])
        neptune.send_metric("val_mean_squared_error", epoch, logs["val_mean_squared_error"])
        
        neptune.send_metric("mean_absolute_error", epoch, logs["mean_absolute_error"])
        neptune.send_metric("val_mean_absolute_error", epoch, logs["val_mean_absolute_error"])
        
        neptune.send_metric("mean_absolute_percentage_error", epoch, logs["mean_absolute_percentage_error"])
        neptune.send_metric("val_mean_absolute_percentage_error", epoch, logs["val_mean_absolute_percentage_error"])
        
        neptune.send_metric("cosine_similarity", epoch, logs["cosine_similarity"])
        neptune.send_metric("val_cosine_similarity", epoch, logs["val_cosine_similarity"])

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    return img

AUTOTUNE    = tf.data.experimental.AUTOTUNE
#image_ds    = tf.data.Dataset.list_files(str(image_dir/'*/*')).map(process_path, num_parallel_calls=AUTOTUNE)
#label_ds    = tf.data.Dataset.from_tensor_slices(labels)
#ds          = tf.data.Dataset.zip((image_ds, label_ds)).shuffle(buffer_size=10) 

#for elem in ds.take(1):
    #print(elem)

points = [(points_pd.iloc[i]['valence'], points_pd.iloc[i]['arousal']) for i in range(img_count)]

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    frames_path, labels=points, class_names=None, seed=123, validation_split=0.2, subset="training", color_mode='rgb', batch_size=batch_size, image_size=(img_height, img_width) 
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    frames_path, labels=points, class_names=None, seed=123, validation_split=0.2, subset="validation", color_mode='rgb', batch_size=batch_size, image_size=(img_height, img_width) 
)

train_ds = train_ds.cache().shuffle(buffer_size=10).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().shuffle(buffer_size=10).prefetch(buffer_size=AUTOTUNE)

# train_ds    = ds.skip(img_count // val_frac).cache().batch(batch_size).shuffle(buffer_size=10).prefetch(buffer_size=AUTOTUNE)
# val_ds      = ds.take(img_count // val_frac).cache().batch(batch_size).shuffle(buffer_size=10).prefetch(buffer_size=AUTOTUNE)

print("=========== Datasets read ==============")

print("Train Dataset Length: ", tf.data.experimental.cardinality(train_ds))
print("Validation Length: ", tf.data.experimental.cardinality(val_ds))

for elem in train_ds.take(1):
    print(elem)

# data_augmentation = keras.Sequential([
#     layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
#     layers.experimental.preprocessing.RandomRotation(0.1),
# ])

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = tf.keras.Sequential([
        # data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(dropout),  
        layers.Flatten(),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(2, activation='tanh')
    ])

neptune.init("Soundbendor/playlist", api_token=api_token)
exp = neptune.create_experiment(params=PARAMS, upload_source_files=["model.py"])
neptune_callback = NeptuneMonitor()

model.compile(
  optimizer='RMSprop',
  loss='cosine_similarity',
  metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_similarity'])

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[neptune_callback]
)

model.summary()
