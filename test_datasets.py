# import necessary modules
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
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

# Some constants good to figure out now
datasheet   = "./msdeezerplus.csv"
samplejson  = "./quadrants/std-22-05-03_1229/songs.json"
samplecount = int(sys.argv[1]) if len(sys.argv) > 1 else 100

# Load random sample points.
samples = {}
while not os.path.exists(samplejson) or not samplejson.endswith(".json"):
    samplejson = input("Sample JSON file not found! Please enter a valid path: ")
with open(samplejson) as f:
    samples = json.load(f)
    print("Sample file loaded!")

# TODO: create random sample pairs
quadrants = ["BL", "BR", "TL", "TR"]
point_combos = {}

# Use itertools to generate quadrant combos:
    # product if we want inner quadrants too
    # permutations if we want both ways
    # comdinations if we only care about one way
quadrant_combos = list(itertools.permutations(quadrants, 2))

print("\nLoading point combos!")
for a, b in quadrant_combos:
    pairname = "{}{}".format(a, b)
    if pairname in point_combos:
        continue
    point_combos[pairname] = []
    print("- {} ... ".format(pairname), end='')

    for i, j in itertools.product(range(samplecount), repeat=2):
        point_combos[pairname].append((int(samples[a][i]), int(samples[b][j])))
    
    print("Loaded!")

# grab column sets for each dataset.
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
datasets = [
    SongDataset(
        name="Deezer",
        cols=cols["deezer"],
        path=datasheet, knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="Deezer + Spotify",
        cols=cols["deezer"] + cols["spotify"],
        path=datasheet, knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="Deezer + MSD",
        cols=cols["deezer"] + cols["msd"],
        path=datasheet, knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="Deezer + Spotify + MSD",
        cols=cols["deezer"] + cols["spotify"] + cols["msd"],
        path=datasheet, knn=True, verbose=True, start_index=3
    )
]

# TODO: PCA and feature selection on MSD and Spotify columns
# Then we can refactor it into the SongDataset object and add to 'datasets'.

# For each dataset and point combination:
for dataset in datasets:
    print()
    print(dataset.name)
    print("length: ", len(dataset))
    dataset.full_df.info(verbose=False)
    dataset.data_df.info(verbose=False)

    # For each point combination:
    for name, pairs in point_combos.items():
        for orig, dest in pairs:

            # TODO: test at various lengths
            length = 10

            # TODO: test at various neighbor counts
            neighbors = 7

            songlist_cos, smoothie_cos, pointlist_cos = prodplay.makePlaylist(
                dataset, orig, dest, length, algos.cosine_score, neighbors
            )
            
            songlist_cos, smoothie_cos, pointlist_cos = prodplay.makePlaylist(
                dataset, orig, dest, length, algos.euclidean_score, neighbors
            )

            # use cosine similarity and euclidean distance
            # use mean squared error for smoothness
            # use variance of step sizes for even steps
            # collect table of results

# grab summmary results
# create relevant graphs