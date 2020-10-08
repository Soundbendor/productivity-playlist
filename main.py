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

#our modules
import helper
import prodplay
import oldplay
import algos
import tests

#get important personal information from Spotify API
info = {}
configFile = sys.argv[1] if (len(sys.argv) > 1) else input("Please enter the path of your config file: ")
while not os.path.exists(configFile) or not configFile.endswith(".json"):
    configFile = input("Config JSON file not found! Please enter a valid path: ")
with open(configFile) as f:
    info = json.load(f)

sp = helper.Spotify(
    info["auth"]["client_id"], 
    info["auth"]["client_secret"], 
    info["auth"]["redirect_uri"], 
    info["auth"]["username"], 
    info["main"]["scope"]
)
songdata = pd.read_csv(
    info["main"]["songdata"], 
    header=0, index_col=0, usecols=[0,3,4,7]
)
has_sp_id = songdata['sp_track_id'] != None
songdata = songdata[has_sp_id]

songpoints = {}
with open(info["main"]["songpoints"]) as f:
    songpoints = json.load(f)

coords = []
for key in songpoints.keys():
    coords.append(helper.string2arrPoint(key))
coords = np.array(coords)

print("N: {}".format(len(songpoints.keys())))
print("Sqrt(N): {}".format(np.sqrt(len(songpoints.keys()))))

# train a KNN model 
model = NearestNeighbors()
model.fit(coords)

# tests.test_neighbors(model, songdata, songpoints, coords)
# tests.test_dists(model, songdata, songpoints, coords)

# user_orig       = 762954
# user_dest       = 1157536
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

# test_time = str(time.strftime("%y-%m-%d_%H%M"))
# helper.makeDir('graph-results/{}'.format(test_time))

# helper.graph('valence', 'arousal', newpoints, data_dim = 2, marker='.',
#     file="graph-results/{}/playlist.png".format(test_time),
#     title = "Example Playlist Path".format(
#         len(newpoints[0]), neighbors,
#         np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
#         np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2), 
#     )
# )

# track_ids = []
# for i in range(len(newsongs)):
#     track_ids.append(songdata.loc[newsongs[i]][2])
# pprint.pprint(track_ids)

# title = "Productivity Playlist Test " + test_time
# helper.makeSpotifyList(sp, username, title, track_ids, False)