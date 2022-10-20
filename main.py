#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import json
import pprint
import time
import sys
import os
import math

#our modules
import helper
import prodplay
import algos
import tests
import spotify
import plot
from songdataset import SongDataset

#get important personal information from Spotify API
info = helper.loadConfig()

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


sp, spo = spotify.Spotify(
    info["auth"]["client_id"], 
    info["auth"]["client_secret"], 
    info["auth"]["redirect_uri"], 
    info["auth"]["username"], 
    info["main"]["scope"]
)

songdata = SongDataset(
    name="Deezer+Spotify",
    cols=cols["deezer"] + cols["spotify"],
    path=info["main"]["songdata"], knn=True, verbose=True,
    data_index = 5, arousal = 4, valence = 3,
)
songdata.make_knn()

pointdata = SongDataset(
    name="Deezer",
    cols=cols["deezer"],
    path=info["main"]["songdata"], knn=True, verbose=True,
    data_index = 3, arousal = 4, valence = 3,
)
pointdata.make_knn()

print("N: {}".format(len(songdata)))
print("Sqrt(N): {}".format(np.sqrt(len(songdata))))

# 3135555   = Daft Punk         - Digital Love          (0.950468991,0.572575336)
# 3135561   = Daft Punk         - Something About Us    (-0.317973857,-0.399224044)
# 540954    = Rachael Yamagata  - 1963                  (1.070081923,1.018911652)
# 533164    = Patty Loveless    - How Can I Help U ...  (-1.636899729,-0.45914527)

user_orig       = 5522768
user_dest       = 3134935
n_songs_reqd    = 10

testsuite = [
    {"name": "points only", "dataset": pointdata},
    {"name": "with songs", "dataset": songdata}
]
testpoints = []

for obj in testsuite:
    name = obj["name"]
    data = obj["dataset"]

    songs, points, feats, smooths, steps = prodplay.makePlaylist(
        data, user_orig, user_dest, n_songs_reqd
    )
    testpoints.append(np.transpose(points))

    print("N Songs Reqd:", n_songs_reqd)
    print("orig:", user_orig, 
            data.full_df.loc[user_orig]['artist_name'], "\t",
            data.full_df.loc[user_orig]['track_name'], "\t",
            "({},{})".format(
                np.around(data.full_df.loc[user_orig]['valence'], decimals=2), 
                np.around(data.full_df.loc[user_orig]['arousal'], decimals=2)
            ))
    print("dest:", user_dest, 
            data.full_df.loc[user_dest]['artist_name'], "\t",
            data.full_df.loc[user_dest]['track_name'], "\t",
            "({},{})".format(
                np.around(data.full_df.loc[user_dest]['valence'], decimals=2), 
                np.around(data.full_df.loc[user_dest]['arousal'], decimals=2)
            ))

    prodplay.printPlaylist(songdata, songs, points, smooths, steps)

helper.graph('valence', 'arousal', testpoints, 
    data_dim = 2, line_count = 2, marker='.',
    legend=[obj["name"] for obj in testsuite],
    file = "newalgo-test/1013-5.png",
    title = "Playlist from {} to {}".format(user_orig, user_dest)
)

# scores = [
#     { "func": algos.cosine_score, "name": "Cosine Similarity"}
#     ,{ "func": algos.euclidean_score, "name": "Euclidean Distance"}
#     ,{ "func": algos.manhattan_score, "name": "Manhattan Distance"}
#     ,{ "func": algos.minkowski3_score, "name": "Minkowski Distance (order 3)"}
#     ,{ "func": algos.jaccard_score, "name": "Jaccard Distance"}
#     ,{ "func": algos.mult_score, "name": "Multiplied Ratios"}
#     ,{ "func": algos.neighbors_rand, "name": "Random Neighbors"}
# ]

# test_dir = "graph-results/{}".format(str(time.strftime("%y-%m-%d_%H%M")))
# helper.makeDir(test_dir)

# for s in scores:
#     newsongs, newpoints, newsmooth, neweven = prodplay.makePlaylist(
#         songdata, user_orig, user_dest, n_songs_reqd, score = s['func']
#     )
#     newpoints = np.transpose(newpoints)
#     newsongs = newsongs.tolist()

#     track_ids = []
#     for song in newsongs:
#         track_ids.append(songdata.get_spid(song))
#     pprint.pprint(track_ids)

#     title = "Playlist {} {}".format(s["name"], str(time.strftime("%Y-%m-%d %H:%M")))
#     helper.makeSpotifyList(sp, spo, title, track_ids, True)

#     helper.graph(
#         "valence", "arousal", newpoints, data_dim=2,
#         title="Example Path using {}".format(s['name']),
#         marker=".",
#         file="{}/{}.png".format(test_dir, s['name'])
#     )

# tests.test_lengths(songdata)
# tests.test_neighbors(songdata)
# tests.test_dists(songdata)