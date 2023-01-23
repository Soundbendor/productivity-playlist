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
import warnings

#our modules
import helper
import prodplay
import spotify
import plot
import algos
import testing
from songdataset import SongDataset
from segmentdataset import SegmentDataset

#get important personal information from Spotify API
datasetpath = "data/deezer/deezer-std-all.csv"
segmentpath = "data/deezer/deezer-segments-dur030.csv"
info = helper.loadConfig("config.json")

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
    feat_index = 5, arousal = 4, valence = 3,
)

pointdata = SongDataset(
    name="Deezer",
    cols=info["cols"]["deezer"],
    path=datasetpath, knn=True, verbose=True,
    feat_index = 3, arousal = 4, valence = 3,
)

segmentdata = SegmentDataset(
    name="Deezer+Segments",
    cols=info["cols"]["deezer"] + info["cols"]["segments"],
    path=segmentpath, knn=True, verbose=True,
    feat_index = 5, arousal = 4, valence = 3,
)

print("N: {}".format(len(segmentdata)))
print("Sqrt(N): {}".format(np.sqrt(len(segmentdata))))

# 3135555   = Daft Punk         - Digital Love          (0.950468991,0.572575336)
# 3135561   = Daft Punk         - Something About Us    (-0.317973857,-0.399224044)
# 540954    = Rachael Yamagata  - 1963                  (1.070081923,1.018911652)
# 533164    = Patty Loveless    - How Can I Help U ...  (-1.636899729,-0.45914527)

user_orig       = 5522768
user_dest       = 3134935
n_songs_reqd    = 10

testsuite = [
    {"name": "points only", "dataset": pointdata},
    {"name": "with features", "dataset": songdata},
    {"name": "with segments", "dataset": segmentdata},
]
testpoints = []
testdir = helper.makeTestDir("main")

print("N Songs Reqd:", n_songs_reqd)
print("orig:", user_orig, 
        pointdata.full_df.loc[user_orig]['artist_name'], "\t",
        pointdata.full_df.loc[user_orig]['track_name'], "\t",
        "({},{})".format(
            np.around(pointdata.full_df.loc[user_orig]['valence'], decimals=2), 
            np.around(pointdata.full_df.loc[user_orig]['arousal'], decimals=2)
        ))
print("dest:", user_dest, 
        pointdata.full_df.loc[user_dest]['artist_name'], "\t",
        pointdata.full_df.loc[user_dest]['track_name'], "\t",
        "({},{})".format(
            np.around(pointdata.full_df.loc[user_dest]['valence'], decimals=2), 
            np.around(pointdata.full_df.loc[user_dest]['arousal'], decimals=2)
        ))

for obj in testsuite:
    name = obj["name"]
    data = obj["dataset"]

    playlistDF = prodplay.makePlaylist(
        data, user_orig, user_dest, n_songs_reqd, verbose = 0
    )
    testpoints.append(playlistDF[["valence", "arousal"]].to_numpy())

    print()
    print(playlistDF.to_string())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        playlistDF.to_latex("{}/{}.tex".format(testdir, name))
        print()
    
    print("Pearson:", testing.pearson(playlistDF))
    print("StepVar:", testing.stepvar(playlistDF))
    
    # # Generate Spotify Playlist.
    # title = "Playlist {} {}".format(name, str(time.strftime("%Y-%m-%d-%H:%M")))
    # spid = spotify.makePlaylist(sp, spo, title, playlistDF["id-spotify"], True)
    # splink = "https://open.spotify.com/playlist/{}".format(spid)
    # print("\nSpotify Playlist: {}".format(splink))

plot.playlist(testpoints, 
    legend=[obj["name"] for obj in testsuite],
    file = "{}/compare-graph.png".format(testdir),
    title = "Playlist from {} to {}".format(user_orig, user_dest)
)