import tensorflow as tf
import pandas as pd
import neptune
import numpy as np
import pprint
import helper

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers


PARAMS = {
    "batch_size"    : 32,
    "img_height"    : 360,
    "img_width"     : 640,
    "dropout"       : 0.2,
    "val_split"     : 0.2,
    "train_epochs"  : 1,
    "test_epochs"   : 1,
    "train_dir"     : "./data/afftrain",
    "test_dir"      : "./data/afftest"
}

dims = ['valence', 'arousal']

def split_dataset(ds):
    ds = ds.unbatch()
    print("Split dataset")
    for elem in ds.take(1): print(elem)
    images, labels = [], []

    for elem in ds:
        (image, label) = elem
        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)

def print_datasets(datasets, frames_path):
    print("=========== Dataset read ==============")
    print("Directory: {}".format(frames_path))
    for idx in range(len(datasets)):
        print("Dataset {} Length:".format(idx), len(datasets[idx]))
        for elem in datasets[idx].take(1): print(elem)
    print("=======================================")

def load_dataset(frames_path, train = True):
    AUTOTUNE        = tf.data.experimental.AUTOTUNE
    points_pd       = pd.read_csv("{}/data.csv".format(frames_path), header=0, usecols=[0, 3, 4], index_col = 0)
    scaler          = StandardScaler()
    points_pd[dims] = scaler.fit_transform(points_pd[dims])
    points          = [(points_pd.iloc[i]['valence'], points_pd.iloc[i]['arousal']) for i in range(len(points_pd))]

    if train:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            frames_path, labels=points, class_names=None, seed=123, 
            validation_split=PARAMS["val_split"], subset="training", color_mode='rgb', 
            batch_size=PARAMS["batch_size"], image_size=(PARAMS["img_height"], PARAMS["img_width"]) 
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            frames_path, labels=points, class_names=None, seed=123, 
            validation_split=PARAMS["val_split"], subset="validation", color_mode='rgb', 
            batch_size=PARAMS["batch_size"], image_size=(PARAMS["img_height"], PARAMS["img_width"]) 
        )

        train_ds = train_ds.cache().shuffle(buffer_size=10).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().shuffle(buffer_size=10).prefetch(buffer_size=AUTOTUNE)
        print_datasets([train_ds, val_ds], frames_path)
        return train_ds, val_ds
    
    else:
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            frames_path, labels=points, class_names=None, seed=123,
            color_mode='rgb', batch_size=PARAMS["batch_size"], image_size=(PARAMS["img_height"], PARAMS["img_width"]) 
        )

        ds = ds.cache().shuffle(buffer_size=10).prefetch(buffer_size=AUTOTUNE)
        print_datasets([ds], frames_path)
        return split_dataset(ds)

train_ds, val_ds = load_dataset(PARAMS["train_dir"], True)
test_images, test_labels  = load_dataset(PARAMS["test_dir"], False)
print(test_images.shape)
print(test_labels.shape)

# data_augmentation = keras.Sequential([
#     layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(PARAMS["img_height"], PARAMS["img_width"], 3)),
#     layers.experimental.preprocessing.RandomRotation(0.1),
# ])

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = tf.keras.Sequential([
        # data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(PARAMS["img_height"], PARAMS["img_width"], 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(PARAMS["img_height"], PARAMS["img_width"], 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(PARAMS["dropout"]),  
        layers.Flatten(),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(2, activation='tanh')
    ])

    api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNThkNWU4ZDQtZWU0Mi00YmQ3LTk2MWMtMTEyNTQ0N2MwOWNiIn0="
    class NeptuneMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            # send logs
            for key in logs.keys():
                neptune.send_metric(key, epoch, logs[key])

            # TODO: run things on test set
            if epoch % PARAMS["test_epochs"] == 0:
                test_preds = model.predict(test_images, batch_size=1)
                print(test_preds.shape, test_labels.shape)
                
                y_true = np.transpose(test_labels)
                y_pred = np.transpose(test_preds)

                for i in range(2):
                    pearson = np.corrcoef(y_true[i], y_pred[i])
                    neptune.send_metric("{}_pearson".format(dims[i]), epoch, pearson)

            return

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
        epochs=PARAMS["train_epochs"],
        callbacks=[neptune_callback]
    )

    model.summary()
