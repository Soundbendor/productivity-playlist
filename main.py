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

#get important personal information from Spotify API
info = helper.loadConfig()

twod = [0,3,6,7]
nd = [0,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

sp = helper.Spotify(
    info["auth"]["client_id"], 
    info["auth"]["client_secret"], 
    info["auth"]["redirect_uri"], 
    info["auth"]["username"], 
    info["main"]["scope"]
)
songdata = pd.read_csv(
    info["main"]["songdata"], 
    header=0, index_col=0, usecols=twod
)
has_sp_id = [(songdata.iloc[i][0] != None) for i in range(len(songdata))]
songdata = songdata[has_sp_id]

# songpoints = {}
# with open(info["main"]["songpoints"]) as f:
#     songpoints = json.load(f)

# coords = []
# for key in songpoints.keys():
#     coords.append(helper.string2arrPoint(key))
# coords = np.array(coords)

coords = []
for i in range(len(songdata)):
    coords.append(songdata.iloc[i][1:].tolist())
coords = np.array(coords)

print("N: {}".format(len(songdata)))
print("Sqrt(N): {}".format(np.sqrt(len(songdata))))

user_orig       = 3135555
user_dest       = 3135561
neighbors       = 10
n_songs_reqd    = 20

# none_count = 0
# nan_count = 0
# inf_count = 0

# for i in range(len(coords)):
#     for j in range(len(coords[i])):
#         if coords[i][j] == None:
#             none_count = none_count + 1
#         if np.isnan(coords[i][j]):
#             nan_count = nan_count + 1
#             print(songdata.iloc[i][0], i, j, coords[i][j], np.dtype(coords[i][j]))
#         if np.isinf(coords[i][j]):
#             inf_count = inf_count + 1

# print("Nones: {}, NaNs: {}, Infs: {}".format(none_count, nan_count, inf_count))

# train a KNN model 
model = NearestNeighbors()
model.fit(coords)

# newsongs, newsmooth, newpoints = prodplay.makePlaylist(
#     songdata, coords, user_orig, user_dest, n_songs_reqd, model, si=1
# )
# newpoints = np.transpose(newpoints)
# labels = ["{} - {}".format(songdata.loc[song][2], songdata.loc[song][3]) for song in newsongs]
# pprint.pprint(labels)

# helper.graph('valence', 'arousal', newpoints, data_dim = 2, marker='.',
#     file="graph-results/playlist.png",
#     title = "Example Playlist Path".format(
#         len(newpoints[0]), neighbors,
#         np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
#         np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2), 
#     )
# )

# track_ids = []
# for i in range(len(newsongs)):
#     track_ids.append(songdata.loc[newsongs[i]][0])
# pprint.pprint(track_ids)
# title = "Genre-Based Playlist"
# helper.makeSpotifyList(sp, info["auth"]["username"], title, track_ids, False)

# tests.test_lengths(model, songdata, coords)
# tests.test_neighbors(model, songdata, coords)
# tests.test_dists(model, songdata, coords)