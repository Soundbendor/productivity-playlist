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
import algos
import tests

#get important personal information from Spotify API
info = helper.loadConfig()

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

