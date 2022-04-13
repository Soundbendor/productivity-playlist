#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
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
from songdataset import SongDataset

#get important personal information from Spotify API
info = helper.loadConfig()

twod = [0,3,6,7]
nd = [0,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

sp, spo = helper.Spotify(
    info["auth"]["client_id"], 
    info["auth"]["client_secret"], 
    info["auth"]["redirect_uri"], 
    info["auth"]["username"], 
    info["main"]["scope"]
)

songdata = SongDataset(
    name="Deezer",
    path=info["main"]["songdata"],
    cols=twod, 
    start_index = 1, 
    spotify=True
)
songdata.make_knn()

print("N: {}".format(len(songdata)))
print("Sqrt(N): {}".format(np.sqrt(len(songdata))))

# 3135555   = Daft Punk         - Digital Love          (0.950468991,0.572575336)
# 3135561   = Daft Punk         - Something About Us    (-0.317973857,-0.399224044)
# 540954    = Rachael Yamagata  - 1963                  (1.070081923,1.018911652)
# 533164    = Patty Loveless    - How Can I Help U ...  (-1.636899729,-0.45914527)

user_orig       = 533164
user_dest       = 540954
neighbors       = 10
n_songs_reqd    = 15

scores = [
    { "func": algos.cosine_score, "name": "Cosine Similarity"}
    ,{ "func": algos.euclidean_score, "name": "Euclidean Distance"}
    ,{ "func": algos.manhattan_score, "name": "Manhattan Distance"}
    ,{ "func": algos.minkowski3_score, "name": "Minkowski Distance (order 3)"}
    ,{ "func": algos.jaccard_score, "name": "Jaccard Distance"}
    ,{ "func": algos.mult_score, "name": "Multiplied Ratios"}
    ,{ "func": algos.neighbors_rand, "name": "Random Neighbors"}
]

test_dir = "graph-results/{}".format(str(time.strftime("%y-%m-%d_%H%M")))
helper.makeDir(test_dir)

for s in scores:
    newsongs, newsmooth, newpoints = prodplay.makePlaylist(
        songdata, user_orig, user_dest, n_songs_reqd, score = s['func']
    )
    newpoints = np.transpose(newpoints)
    newsongs = newsongs.tolist()

    track_ids = []
    for song in newsongs:
        track_ids.append(songdata.get_spid(song))
    pprint.pprint(track_ids)

    title = "Playlist {} {}".format(s["name"], str(time.strftime("%Y-%m-%d %H:%M")))
    helper.makeSpotifyList(sp, spo, title, track_ids, True)

    helper.graph(
        "valence", "arousal", newpoints, data_dim=2,
        title="Example Path using {}".format(s['name']),
        marker=".",
        file="{}/{}.png".format(test_dir, s['name'])
    )

# tests.test_lengths(songdata)
# tests.test_neighbors(songdata)
# tests.test_dists(songdata)