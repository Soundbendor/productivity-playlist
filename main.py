#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors

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
songdata = pd.read_csv("data_with_features.csv", header=0, index_col=0, usecols=[0, 1, 2, 5])
has_sp_id = songdata['sp_track_id'] != "NO TRACK FOUND ON SPOTIFY"
songdata = songdata[has_sp_id]
song_ids = list(songdata.index.values)

# train a KNN model 
model = NearestNeighbors()
model.fit(songdata.select_dtypes(include='float64').to_numpy())

tests.test_length(model, songdata)