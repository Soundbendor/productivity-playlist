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
datasheet   = "./deezer-std.csv"
samplejson  = "./quadrants/std-22-05-03_1229/songs.json"
samplecount = int(sys.argv[1]) if len(sys.argv) > 1 else 100

# set up output directories
dirname = "./ismir-test/{}".format(str(time.strftime("%y-%m-%d_%H%M")))
helper.makeDir("./ismir-test")
helper.makeDir(dirname)

# Load random sample points.
samples = {}
while not os.path.exists(samplejson) or not samplejson.endswith(".json"):
    samplejson = input("Sample JSON file not found! Please enter a valid path: ")
with open(samplejson) as f:
    samples = json.load(f)
    print("Sample file loaded!")

# create random sample pairs
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
        name="Deezer+Spotify",
        cols=cols["deezer"] + cols["spotify"],
        path=datasheet, knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="Deezer+MSD",
        cols=cols["deezer"] + cols["msd"],
        path=datasheet, knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="Deezer+Spotify+MSD",
        cols=cols["deezer"] + cols["spotify"] + cols["msd"],
        path=datasheet, knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="PCA-Deezer+Spotify",
        path="deezerpca-spotify.csv", 
        knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="PCA-Deezer+MSD",
        path="deezerpca-msd.csv", 
        knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="PCA-Deezer+Spotify+MSD",
        path="deezerpca-spotify+msd.csv", 
        knn=True, verbose=True, start_index=3
    )
]

# TODO?: test at various lengths
length = 10

# TODO?: test at various neighbor counts
neighbors = 7

# Columns for our result sheets
base_cols = ["oq", "dq", "orig", "dest"] 
score_cols = ["cos_smooth", "cos_even", "euc_smooth", "euc_even"]
cos_cols = ["cos_{}".format(i) for i in range(length)]
euc_cols = ["euc_{}".format(i) for i in range(length)]
resultcols = base_cols + score_cols + cos_cols + euc_cols

# For each dataset and point combination:
for dataset in datasets:
    print("Testing {}".format(dataset.name))
    helper.makeDir("{}/{}".format(dirname, dataset.name))

    # collect table of results
    results = {}
    for col in resultcols: results[col] = []

    # For each point combination:
    for oq, dq in quadrant_combos:
        qc = "{}{}".format(oq, dq)
        pairs = point_combos[qc]
        print(" - {}".format(qc))

        curdirname = "{}/{}/{}".format(dirname, dataset.name, qc)
        helper.makeDir(curdirname)

        # use cosine similarity and euclidean distance
        # use mean squared error for smoothness
        # use variance of step sizes for even steps
        for orig, dest in pairs:

            # Generate playlists with each distance
            cos_songs, cos_points, cos_smooth, cos_even = prodplay.makePlaylist(
                dataset, orig, dest, length, algos.cosine_score, neighbors)
            euc_songs, euc_points, euc_smooth, euc_even = prodplay.makePlaylist(
                dataset, orig, dest, length, algos.euclidean_score, neighbors)

            # Graph playlists if we want:
            cos_points = np.transpose(cos_points)
            euc_points = np.transpose(euc_points)
            points = [cos_points, euc_points]
            origPoint = dataset.data_df.loc[orig]
            destPoint = dataset.data_df.loc[dest]

            helper.graph('valence', 'arousal', points, 
                data_dim = 2, line_count = 2, marker='.',
                legend=['Cosine Similarity', 'Euclidean Distance'],
                file = "{}/{}-{}.png".format(curdirname, orig, dest),
                title = "Playlists from {} ({}, {}) to {} ({}, {})".format(
                    orig,
                    np.around(origPoint[0], decimals=2), 
                    np.around(origPoint[1], decimals=2), 
                    dest,
                    np.around(destPoint[0], decimals=2), 
                    np.around(destPoint[1], decimals=2), 
                )
            )

            # We have this problem where (cosine) gets to the destination faster btw. 

            # Add results to our collection
            results["oq"].append(oq)
            results["dq"].append(dq)
            results["orig"].append(orig)
            results["dest"].append(dest)

            results["cos_smooth"].append(cos_smooth)
            results["cos_even"].append(cos_even)
            results["euc_smooth"].append(euc_smooth)
            results["euc_even"].append(euc_even)

            for idx in range(length):
                results["cos_{}".format(idx)].append(
                    cos_songs[idx] if idx < len(cos_songs) else float(0))
                results["euc_{}".format(idx)].append(
                    euc_songs[idx] if idx < len(euc_songs) else float(0))
        
    # Convert table of results into a DataFrame
    results[qc] = pd.DataFrame(results)
    results[qc].to_csv("{}/{}/results.csv".format(dirname, dataset.name))


# grab summmary results
# create relevant graphs