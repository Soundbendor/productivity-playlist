#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import time
import pprint

#our modules
import helper
import prodplay
import algos

#get important personal information from Spotify API
client_id = '2a285d92069147f8a7e59cec1d0d9bb6'
client_secret = '1eebc7035f74489db8f5597ce4afb863'
redirect_uri = 'https://www.google.com/'
username = 'eonkid46853'
scope = "playlist-modify-public,playlist-modify-private"
sp = helper.Spotify(client_id, client_secret, redirect_uri, username, scope)

# load song_id, average arousal and valence, and spotify track ID for each song from CSV to pandas DataFrame
songdata = pd.read_csv("data_with_features.csv", header=0, index_col=0, usecols=[0, 1, 2, 5])
has_sp_id = songdata['sp_track_id']!="NO TRACK FOUND ON SPOTIFY"
songdata = songdata[has_sp_id]
song_ids = list(songdata.index.values)

# train a KNN model 
neigh = NearestNeighbors()
neigh.fit(songdata.select_dtypes(include='float64').to_numpy())

# input the starting and destination coordinates, set "current" to starting
user_curr = song_ids[random.randint(0, len(song_ids)) - 1]
user_dest = song_ids[random.randint(0, len(song_ids)) - 1]
print(songdata.loc[user_curr])
print(songdata.loc[user_dest])

# input the time needed to get from starting to destination (test cases here)
testcount = int(input("How many tests do you want to do? "))
teststart = 2

scores = [algos.euclid_score, algos.add_score, algos.mult_score]
key = ["euclidean distance", "added differences", "multiplied ratios"]
smoothnesses = []

for score in scores:
    smoothies = []      # smoothness values for each collective playlist
    songlists = []      # the different playlists

    # for loop for testing different amounts of points in between
    for n_songs_reqd in range(teststart, teststart + testcount):
        songlist, smoothie = prodplay.makePlaylist(
            songdata, user_curr, user_dest, n_songs_reqd, neigh, score
        )
        
        smoothies.append(smoothie)
        print("{}: {}".format(n_songs_reqd, smoothie))
        songlists.append(songlist)

    coords = []
    for j in range(len(songlists)):
        v_points = []
        a_points = []

        for i in range(len(songlists[j])):
            v_points.append(songdata.loc[songlists[j][i]][1])
            a_points.append(songdata.loc[songlists[j][i]][0])

        wrapped_points = [v_points, a_points]
        coords.append(wrapped_points)

    smoothest = np.argmin(smoothies)
    print(smoothest + teststart)

    helper.graph('valence', 'arousal', coords, 2, len(coords))
    helper.graph('valence', 'arousal', coords[smoothest], 2)
    
    smoothnesses.append(smoothies)

helper.graph('# of tests', 'smoothness of paths', smoothnesses, 1, 3, key)


# track_ids = []
# for i in range(len(songlists[smoothest])):
#     track_ids.append(songdata.loc[songlists[smoothest][i]][2])
# print(track_ids)

# title = "Productivity Playlist Test " + str(time.ctime())
# helper.makeSpotifyList(sp, username, title, track_ids, False)