#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import time
import matplotlib.pyplot as plt
import pprint

import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

#get important personal information from Spotify API
client_id = '2a285d92069147f8a7e59cec1d0d9bb6'
client_secret = '1eebc7035f74489db8f5597ce4afb863'
redirect_uri = 'https://www.google.com/'
username = 'eonkid46853'

#get yo Spotify
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
scope = "playlist-modify-public,playlist-modify-private"
try:
    token = util.prompt_for_user_token(username, scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
    sp = spotipy.Spotify(auth= token)
except:
    print('Token is not accessible for ' + username)

sp = spotipy.Spotify(auth=token, client_credentials_manager=client_credentials_manager)

# load song_id, average arousal and valence for each song from CSV to pandas DataFrame
# songdata = pd.read_csv("deam-data\\annotations\\annotations averaged per song\song_level\static_annotations_averaged_songs_1_2000.csv", header=0, index_col=0, usecols=[0, 1, 3])
# songdata = pd.read_csv("deezer-data\\train.csv", header=0, index_col=0, usecols=[0, 3, 4])
songdata = pd.read_csv("data_with_features.csv", header=0, index_col=0, usecols=[0, 1, 2, 5])
has_sp_id = songdata['sp_track_id']!="NO TRACK FOUND ON SPOTIFY"
songdata = songdata[has_sp_id]
print("read song data")
song_ids = list(songdata.index.values)

# train a KNN model 
neigh = NearestNeighbors()
neigh.fit(songdata.select_dtypes(include='float64').to_numpy())
print("trained data ... fingers crossed")

# input the starting and destination coordinates, set "current" to starting
# user_curr = int(input("Choose a number for a starting song: "))
# user_dest = int(input("Choose a number for an ending song: "))
user_curr = song_ids[random.randint(0, len(song_ids)) - 1]
user_dest = song_ids[random.randint(0, len(song_ids)) - 1]
print(songdata.loc[user_curr])
print(songdata.loc[user_dest])

# input the time needed to get from starting to destination (test cases here)
# testcount = int(input("How many songs between your start and destination? "))
testcount = int(input("How many tests do you want to do? "))
teststart = 2
# testcount = 100

# array of smoothness values
smoothies = []

# array to store coordinates for all playlists
coords = []

#array to store the Deezer Song IDs for the smoothest playlist (for Spotify output)
min_id_list = []
smooth = (1, 99999)

# for loop for testing different amounts of points in between
for n_songs_reqd in range(teststart, teststart + testcount):
    current = user_curr
    destination = user_dest
    origin_a = songdata.loc[current][1]
    origin_v = songdata.loc[current][0]
    destination_a = songdata.loc[destination][1]
    destination_v = songdata.loc[destination][0]

    sm_a_step = (destination_a - origin_a) / n_songs_reqd
    sm_v_step = (destination_v - origin_v) / n_songs_reqd

    # create a list of numbers (for songs) to store our "path" - let's call it song_list
    songlist = np.empty(0)
    songlist = np.append(songlist, current)

    # testing variable for running a single iteration of the while loop
    test = True 
    
    #storing the "smoothness" values for each song in the playlist
    smooth_steps = []

    # while the current song isn't the destination song and the number of songs required isn't met
    while ((current != destination) & (len(songlist) < n_songs_reqd)):
    # while (test):
        # grab the a_dist and v_dist between current and destination (replace with list for scaling dimensions)
        current_a = songdata.loc[current][1]
        current_v = songdata.loc[current][0]

        # was gonna do slope and distance, but what if the imagined vector was in the OPPOSITE direction?
        distance = np.sqrt(np.square(destination_a - current_a) + np.square(destination_v - current_v))
        dist_a = destination_a - current_a + .001
        dist_v = destination_v - current_v + .001

        # divide distance by the n_songs_reqd to get a step size
        step_size = distance / (n_songs_reqd - len(songlist))
        step_a = dist_a / (n_songs_reqd - len(songlist))
        step_v = dist_v / (n_songs_reqd - len(songlist))

        # grab the nearest neighbors within 110% * step_size of the current point
        r_neighbors = neigh.radius_neighbors([songdata.select_dtypes(include='float64').loc[current].array], radius=(1.1 * step_size))
        candidates = np.array(r_neighbors[1])
        candidates = candidates[0]

        for i in range(len(songlist)):
            candidates = candidates[candidates != songdata.index.get_loc(songdata.loc[songlist[i]].name)]
        
        if (len(candidates) < 1):
            k_neighbors = neigh.kneighbors([songdata.select_dtypes(include='float64').loc[current].array], n_songs_reqd + 1)
            candidates = np.array(k_neighbors[1])
            candidates = candidates[0]

            for i in range(len(songlist)):
                candidates = candidates[candidates != songdata.index.get_loc(songdata.loc[songlist[i]].name)]

        cand_scores = []
        cand_smooths = []

        # TEST ALL THE CANDIDATES FOR THE NEXT SONG TO FIND THE BEST ONE
        for i in range(len(candidates)):
            num = candidates[i]
            
            ## RATIO METHOD
            # dist_a_ratio = (songdata.iloc[num][1] - current_a) / step_a
            # dist_v_ratio = (songdata.iloc[num][0] - current_v) / step_v
            # cand_scores.append(np.absolute(1 - dist_a_ratio) * np.absolute(1 - dist_v_ratio))

            ## DIFFERENCE METHOD
            dist_a_diff = songdata.iloc[num][1] - current_a - step_a
            dist_v_diff = songdata.iloc[num][0] - current_v - step_v

            smooth_a_diff = songdata.iloc[num][1] - (origin_a + ((len(songlist) - 1) * sm_a_step))
            smooth_v_diff = songdata.iloc[num][0] - (origin_v + ((len(songlist) - 1) * sm_v_step))
            
            added_dist = dist_a_diff + dist_v_diff
            euclid_dist = np.sqrt(np.square(dist_a_diff) + np.square(dist_v_diff))
            euclid_smooth = np.sqrt(np.square(smooth_a_diff) + np.square(smooth_v_diff))

            cand_scores.append(euclid_dist)
            cand_smooths.append(euclid_smooth)

        # select the song which has the ratio/difference closest to 1.00 to be the new value of "current"
        min_indices = np.argmin(cand_scores)
        min_cand_smooth = cand_smooths[min_indices]
        smooth_steps.append(min_cand_smooth)

        min_cand_index = candidates[min_indices]
        current = song_ids[min_cand_index]
        songlist = pd.unique(np.append(songlist, current))
        test = False

    smoothie = np.mean(smooth_steps)

    if (len(smoothies) > 0):
        if (smoothie < smoothies[np.argmin(smoothies)]):
            min_id_list = songlist

    smoothies.append(smoothie)
    print("{}: {}".format(n_songs_reqd, smoothie))

    v_points = []
    a_points = []

    for i in range(len(songlist)):
        v_points.append(songdata.loc[songlist[i]][1])
        a_points.append(songdata.loc[songlist[i]][0])

    wrapped_points = [v_points, a_points]
    coords.append(wrapped_points)

plt.xlabel('valence')
plt.ylabel('arousal')
plt.legend(smoothies)

smoothest = np.argmin(smoothies)
print(smoothest + teststart)

for i in range(len(coords)):
    plt.plot(coords[i][0], coords[i][1])
plt.show()

plt.xlabel('valence')
plt.ylabel('arousal')
plt.plot(coords[smoothest][0], coords[smoothest][1])
plt.show()

plt.xlabel('# of tests')
plt.ylabel('smoothness of paths')
plt.plot(smoothies)
plt.show()

title = "Productivity Playlist Test " + str(time.ctime())
result_playlist = sp.user_playlist_create(username, title, public=False)

track_ids = []
for i in range(len(min_id_list)):
    track_ids.append(songdata.loc[min_id_list[i]][2])

print(track_ids)
sp.user_playlist_add_tracks(username, result_playlist['id'], track_ids)