#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import json
import pprint

#our modules
import helper
import prodplay
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
# has_sp_id = songdata['has'] != None
# songdata = songdata[has_sp_id]

print("N: {}".format(len(songdata)))
print("Sqrt(N): {}".format(np.sqrt(len(songdata))))

songpoints = {}
with open("songpoints.json") as f:
    songpoints = json.load(f)

coords = []
for key in songpoints.keys():
    coords.append(helper.string2arrPoint(key))
coords = np.array(coords)
pprint.pprint(coords)
pprint.pprint(songdata.select_dtypes(include='float64').to_numpy())

# train a KNN model 
model = NearestNeighbors()
model.fit(coords)

tests.test_neighbors(model, songdata, songpoints, coords)