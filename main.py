#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import json
import pprint

#our modules
import helper
import prodplay
import oldplay
import algos
import tests

#get important personal information from Spotify API
client_id = '2a285d92069147f8a7e59cec1d0d9bb6'
client_secret = '1eebc7035f74489db8f5597ce4afb863'
redirect_uri = 'https://www.google.com/'
username = 'eonkid46853'
scope = "playlist-modify-public,playlist-modify-private"
sp = helper.Spotify(client_id, client_secret, redirect_uri, username, scope)

# load song_id, average arousal and valence, and spotify track ID for each song from CSV to pandas DataFrame
songdata = pd.read_csv("deezer-spotify.csv", header=0, index_col=0, usecols=[0, 3, 4, 7])
# songdata = pd.read_csv("deam-data/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv", header=0, index_col=0, usecols=[0, 1, 3])
has_sp_id = songdata['sp_track_id'] != None
songdata = songdata[has_sp_id]

songpoints = {}
with open("songpoints.json") as f:
    songpoints = json.load(f)

coords = []
for key in songpoints.keys():
    coords.append(helper.string2arrPoint(key))
coords = np.array(coords)

print("N: {}".format(len(songpoints.keys())))
print("Sqrt(N): {}".format(np.sqrt(len(songpoints.keys()))))

plotCoords = np.transpose(coords)
plt.scatter(plotCoords[0], plotCoords[1], marker='.', linewidths=0.7, c=["#D73F09"])
plt.show()

# train a KNN model 
model = NearestNeighbors()
model.fit(coords)

# tests.test_neighbors(model, songdata, songpoints, coords)
# tests.test_dists(model, songdata, songpoints, coords)

# user_orig       = 532216
# user_dest       = 532284
# neighbors       = 10
# n_songs_reqd    = 20

# print(songdata.loc[user_orig])
# print(songdata.loc[user_dest])

# print("\n\nNEW method:")
# newsongs, newsmooth, newpoints = prodplay.makePlaylist(
#     songdata, songpoints, coords, 
#     user_orig, user_dest, n_songs_reqd, 
#     model
# )
# pprint.pprint(newsongs)
# pprint.pprint(newsmooth)
# newpoints = np.transpose(newpoints)
# pprint.pprint(newpoints)

# print("\n\nOLD method:")
# oldsongs, oldsmooth = oldplay.makePlaylist(
#     songdata, songpoints, coords, 
#     user_orig, user_dest, n_songs_reqd, 
#     model
# )
# pprint.pprint(oldsongs)
# pprint.pprint(oldsmooth)

# helper.graph('valence', 'arousal', newpoints, data_dim = 2, marker='.',
#     file="graph-results/new.png",
#     title = "Path ({} songs, K={}) from ({}, {}) to ({}, {})".format(
#         len(newpoints[0]), neighbors,
#         np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
#         np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2), 
#     )
# )

# oldpoints = [[],[]]
# for song in oldsongs:
#     oldpoints[0].append(songdata.loc[song][0])
#     oldpoints[1].append(songdata.loc[song][1])
# pprint.pprint(oldpoints)

# helper.graph('valence', 'arousal', oldpoints, data_dim = 2, marker='.',
#     file="graph-results/old.png",
#     title = "Path ({} songs, K={}) from ({}, {}) to ({}, {})".format(
#         len(oldpoints[0]), neighbors,
#         np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
#         np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2), 
#     )
# )