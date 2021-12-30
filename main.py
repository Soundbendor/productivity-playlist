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

# newsongs, newsmooth, newpoints = prodplay.makePlaylist(
#     songdata, user_orig, user_dest, n_songs_reqd
# )
# newpoints = np.transpose(newpoints)
# newsmooth = [0] + newsmooth.tolist() + [0]
# newsongs = newsongs.tolist()

# fullsongdata = pd.read_csv(info["main"]["songdata"], header=0, index_col=0)
# songtitles = [fullsongdata.loc[s]["track_name"] for s in newsongs]
# songartists = [fullsongdata.loc[s]["artist_name"] for s in newsongs]

# jsondata = {
#     "id": newsongs,
#     "track": songtitles,
#     "artist": songartists,
#     "valence": newpoints[0].tolist(),
#     "arousal": newpoints[1].tolist(),
#     "smooth": newsmooth
# }

# pprint.pprint(jsondata)

# csv_df = pd.DataFrame(jsondata)
# csv_df.to_csv("expoints.csv")

# newpoints = np.transpose(newpoints)
# labels = ["{} - {}".format(songdata.loc[song][2], songdata.loc[song][3]) for song in newsongs]
# pprint.pprint(labels)

# track_ids = []
# for i in range(len(newsongs)):
#     track_ids.append(songdata.loc[newsongs[i]][0])
# pprint.pprint(track_ids)
# title = "Playlist {}".format(str(time.strftime("%Y-%m-%d %H:%M")))
# helper.makeSpotifyList(sp, spo, title, track_ids, True)

# mse_annotations = ["{:.2e}".format(p) for p in newsmooth]
# helper.graph('valence', 'arousal', newpoints, data_dim=2, marker='.', title='Example Playlist', file='exgraph.png', av_circle=True, point_annotations=mse_annotations)


# tests.test_lengths(songdata)
# tests.test_neighbors(songdata)
tests.test_dists(songdata)