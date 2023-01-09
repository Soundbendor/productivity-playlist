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
from songdataset import SongDataset

#get important personal information from Spotify API
info = helper.loadConfig("config.json")
sp, spo = spotify.Spotify(
    info["auth"]["client_id"], 
    info["auth"]["client_secret"], 
    info["auth"]["redirect_uri"], 
    info["auth"]["username"], 
    info["auth"]["scope"]
)

print(sp.me())

songdata = SongDataset(
    name="Deezer+Spotify",
    cols=info["cols"]["deezer"] + info["cols"]["spotify"],
    path="data/deezer/deezer-std-all.csv", knn=True, verbose=True,
    data_index = 5, arousal = 4, valence = 3,
)

user_orig       = 5522768
user_dest       = 3134935
n_songs_reqd    = 10

playlistDF = prodplay.makePlaylist(
    songdata, user_orig, user_dest, n_songs_reqd, verbose = 0
)

print()
print(playlistDF.to_string())
    
track_ids = playlistDF["id-spotify"].tolist()
title = "Playlist {} {}".format("Test", str(time.strftime("%Y-%m-%d-%H:%M")))
spotify_id = spotify.makePlaylist(sp, spo, title, track_ids, True)
print("\nSpotify Playlist ID: {}".format(spotify_id))