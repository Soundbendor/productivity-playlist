# import necessary modules
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
from pprint import pprint
import time
import sys
import os
import math
import itertools

#our modules
import helper
import prodplay
import algos
import tests
from songdataset import SongDataset

datasheet = "./msdeezer-std.csv"
dirname = "./deezer-analysis/pca"
helper.makeDir(dirname)
helper.makeDir("{}/charts".format(dirname))

cols = {
    "deezer": [
        "dzr_sng_id",
        "sp_track_id",
        "artist_name",
        "track_name",
        "valence",
        "arousal"
    ],
    "spotify": [
        "sp_acousticness",
        "sp_danceability",
        "sp_duration_ms",
        "sp_energy",
        "sp_instrumentalness",
        "sp_key",
        "sp_liveness",
        "sp_loudness",
        "sp_mode",
        "sp_speechiness",
        "sp_tempo",
        "sp_time_sig",
        "sp_valence",
        "sp_popularity",
        "sp_explicit"
    ],
    "msd": [
        # "MSD_artist_familiarity","MSD_artist_hotttnesss",
        # "MSD_danceability","MSD_energy",
        "MSD_key","MSD_key_confidence","MSD_loudness",
        "MSD_mode","MSD_mode_confidence","MSD_tempo",
        "MSD_time_signature","MSD_time_signature_confidence",
        "MSD_segments_loudness_max_avg","MSD_segments_loudness_max_std","MSD_segments_loudness_max_var","MSD_segments_loudness_max_min","MSD_segments_loudness_max_max","MSD_segments_loudness_max_med",
        "MSD_segments_loudness_max_time_avg","MSD_segments_loudness_max_time_std","MSD_segments_loudness_max_time_var","MSD_segments_loudness_max_time_min","MSD_segments_loudness_max_time_max","MSD_segments_loudness_max_time_med",
        "MSD_segments_timbre_0_avg","MSD_segments_timbre_0_std","MSD_segments_timbre_0_var","MSD_segments_timbre_0_min","MSD_segments_timbre_0_max","MSD_segments_timbre_0_med",
        "MSD_segments_timbre_1_avg","MSD_segments_timbre_1_std","MSD_segments_timbre_1_var","MSD_segments_timbre_1_min","MSD_segments_timbre_1_max","MSD_segments_timbre_1_med",
        "MSD_segments_timbre_2_avg","MSD_segments_timbre_2_std","MSD_segments_timbre_2_var","MSD_segments_timbre_2_min","MSD_segments_timbre_2_max","MSD_segments_timbre_2_med",
        "MSD_segments_timbre_3_avg","MSD_segments_timbre_3_std","MSD_segments_timbre_3_var","MSD_segments_timbre_3_min","MSD_segments_timbre_3_max","MSD_segments_timbre_3_med",
        "MSD_segments_timbre_4_avg","MSD_segments_timbre_4_std","MSD_segments_timbre_4_var","MSD_segments_timbre_4_min","MSD_segments_timbre_4_max","MSD_segments_timbre_4_med",
        "MSD_segments_timbre_5_avg","MSD_segments_timbre_5_std","MSD_segments_timbre_5_var","MSD_segments_timbre_5_min","MSD_segments_timbre_5_max","MSD_segments_timbre_5_med",
        "MSD_segments_timbre_6_avg","MSD_segments_timbre_6_std","MSD_segments_timbre_6_var","MSD_segments_timbre_6_min","MSD_segments_timbre_6_max","MSD_segments_timbre_6_med",
        "MSD_segments_timbre_7_avg","MSD_segments_timbre_7_std","MSD_segments_timbre_7_var","MSD_segments_timbre_7_min","MSD_segments_timbre_7_max","MSD_segments_timbre_7_med",
        "MSD_segments_timbre_8_avg","MSD_segments_timbre_8_std","MSD_segments_timbre_8_var","MSD_segments_timbre_8_min","MSD_segments_timbre_8_max","MSD_segments_timbre_8_med",
        "MSD_segments_timbre_9_avg","MSD_segments_timbre_9_std","MSD_segments_timbre_9_var","MSD_segments_timbre_9_min","MSD_segments_timbre_9_max","MSD_segments_timbre_9_med",
        "MSD_segments_timbre_10_avg","MSD_segments_timbre_10_std","MSD_segments_timbre_10_var","MSD_segments_timbre_10_min","MSD_segments_timbre_10_max","MSD_segments_timbre_10_med",
        "MSD_segments_timbre_11_avg","MSD_segments_timbre_11_std","MSD_segments_timbre_11_var","MSD_segments_timbre_11_min","MSD_segments_timbre_11_max","MSD_segments_timbre_11_med"
    ]
}

# Let's create an array of the song datasets.
print("\nLoading datasets.")
base = SongDataset(
    name="Deezer",
    cols=cols["deezer"],
    path=datasheet, knn=True, verbose=True, start_index=3
)

datasets = [
    SongDataset(
        name="spotify",
        cols=cols["deezer"] + cols["spotify"],
        path=datasheet, knn=True, verbose=True, start_index=5
    ),
    SongDataset(
        name="msd",
        cols=cols["deezer"] + cols["msd"],
        path=datasheet, knn=True, verbose=True, start_index=5
    ),
    SongDataset(
        name="spotify+msd",
        cols=cols["deezer"] + cols["spotify"] + cols["msd"],
        path=datasheet, knn=True, verbose=True, start_index=5
    )
]

for dataset in datasets:
    nc, Xpca = helper.find_nc_PCA(
        dataset, 12, "{}/charts/{}.png".format(dirname, dataset.name))
    
    pca_cols = np.transpose(Xpca)
    basecopy = base.full_df.copy()

    for i in range(12):
        basecopy['pca_{}'.format(i)] = pca_cols[i].tolist()
    
    basecopy.to_csv("{}/deezerpca-{}.csv".format(dirname, dataset.name))
    