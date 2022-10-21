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
import spotify
import plot
import algos
from songdataset import SongDataset

#get important personal information from Spotify API
datasetpath = "data/deezer/deezer-std-all.csv"
info = helper.loadConfig()

scores = [
    { "func": algos.cosine_score, "name": "Cosine Similarity"}
    ,{ "func": algos.euclidean_score, "name": "Euclidean Distance"}
    ,{ "func": algos.manhattan_score, "name": "Manhattan Distance"}
    ,{ "func": algos.minkowski3_score, "name": "Minkowski Distance (order 3)"}
    ,{ "func": algos.jaccard_score, "name": "Jaccard Distance"}
    ,{ "func": algos.mult_score, "name": "Multiplied Ratios"}
]

sp, spo = spotify.Spotify(
    info["auth"]["client_id"], 
    info["auth"]["client_secret"], 
    info["auth"]["redirect_uri"], 
    info["auth"]["username"], 
    info["auth"]["scope"]
)

songdata = SongDataset(
    name="Deezer+Spotify",
    cols=info["cols"]["deezer"] + info["cols"]["spotify"],
    path=datasetpath, knn=True, verbose=True,
    data_index = 5, arousal = 4, valence = 3,
)
songdata.make_knn()

pointdata = SongDataset(
    name="Deezer",
    cols=info["cols"]["deezer"],
    path=datasetpath, knn=True, verbose=True,
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
        data, user_orig, user_dest, n_songs_reqd, verbose = 1
    )
    testpoints.append(points)

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
    
    # track_ids = [songdata.get_spid(s) for s in songs]
    # title = "Playlist {} {}".format(name, str(time.strftime("%Y-%m-%d-%H:%M")))
    # spotify_id = spotify.makePlaylist(sp, spo, title, track_ids, True)
    # print("\nSpotify Playlist ID: {}".format(spotify_id))

plot.playlist(testpoints, 
    legend=[obj["name"] for obj in testsuite],
    file = "out/new-v-old.png",
    title = "Playlist from {} to {}".format(user_orig, user_dest)
)