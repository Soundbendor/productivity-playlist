import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import time
import pprint
import os

import algos
import helper
import prodplay

def test_neighbors(model, songdata):
    song_ids = list(songdata.index.values)
    # input the starting and destination coordinates, set "current" to starting
    user_orig = song_ids[random.randint(0, len(song_ids)) - 1]
    user_dest = song_ids[random.randint(0, len(song_ids)) - 1]
    # user_orig = 239138
    # user_dest = 286183
    print(songdata.loc[user_orig])
    print(songdata.loc[user_dest])

    # input the time needed to get from starting to destination (test cases here)
    testcount = int(input("How many tests do you want to do? "))
    teststart = 2
    
    smoothnesses = []

    neighbor_counts = [15, 17, 19, 21, 23, 25, 27, 29, 31]

    test_time = str(time.strftime("%y-%m-%d_%H%M"))

    if not os.path.exists('graph-results/{}'.format(test_time)):
        os.makedirs('graph-results/{}'.format(test_time))

    for i in range(len(neighbor_counts)):
        print("\n\n{} Neighbors".format(neighbor_counts[i]))
        smoothies = []      # smoothness values for each collective playlist
        songlists = []      # the different playlists

        # for loop for testing different amounts of points in between
        for n_songs_reqd in range(teststart, teststart + testcount):
            songlist, smoothie = prodplay.makePlaylist(
                songdata, user_orig, user_dest, n_songs_reqd, model, neighbors=neighbor_counts[i]
            ) 
            
            smoothies.append(smoothie)
            print("{}: {}".format(n_songs_reqd, smoothie))
            songlists.append(songlist)

        coords = []
        for j in range(len(songlists)):
            v_points = []
            a_points = []

            for k in range(len(songlists[j])):
                v_points.append(songdata.loc[songlists[j][k]][1])
                a_points.append(songdata.loc[songlists[j][k]][0])

            wrapped_points = [v_points, a_points]

            # helper.graph('valence', 'arousal', wrapped_points, data_dim = 2, marker='.',
            #     file = 'graph-results/{}/result_{}_{}.png'.format(test_time, j + teststart, keys[i]),
            #     title = "Playlist Path ({} songs) using {}".format(j + teststart, keys[i])
            # ) 

            coords.append(wrapped_points)

        smoothest = np.argmin(smoothies)
        print(smoothest + teststart)

        helper.graph('valence', 'arousal', coords, data_dim = 2, line_count = len(coords),
            file = 'graph-results/{}/all_lines_{}.png'.format(test_time, neighbor_counts[i]),
            title = "All Playlist Paths (from {} to {} songs) using {} Neighbors".format(
                teststart, teststart + testcount - 1, neighbor_counts[i])
        )
        helper.graph('valence', 'arousal', coords[smoothest], data_dim = 2, marker='.',
            file = 'graph-results/{}/smoothest_{}.png'.format(test_time, neighbor_counts[i]),
            title = "Smoothest Playlist Path ({} songs) using {} Neighbors".format(smoothest + teststart, neighbor_counts[i])
        )
        
        # # PUT THE "smoothest path" ON A SPOTIFY PLAYLIST
        # track_ids = []
        # for i in range(len(songlists[smoothest])):
        #     track_ids.append(songdata.loc[songlists[smoothest][i]][2])
        # print(track_ids)

        # title = "Productivity Playlist Test " + test_time
        # helper.makeSpotifyList(sp, username, title, track_ids, False)

        smoothnesses.append(smoothies)

    helper.graph('length of playlist', 'smoothness of path', smoothnesses, 
        data_dim = 1, line_count = len(smoothnesses), legend = neighbor_counts,
        file = 'graph-results/{}/comparison.png'.format(test_time),
        title = "Mean Squared Error of Playlists Using Different K-Neighbors"
    )

def test_dists(model, songdata):
    song_ids = list(songdata.index.values)
    # input the starting and destination coordinates, set "current" to starting
    user_orig = song_ids[random.randint(0, len(song_ids)) - 1]
    user_dest = song_ids[random.randint(0, len(song_ids)) - 1]
    # user_orig = 239138
    # user_dest = 286183
    print(songdata.loc[user_orig])
    print(songdata.loc[user_dest])

    # input the time needed to get from starting to destination (test cases here)
    testcount = int(input("Maximum playlist length: "))
    teststart = 2
    
    scores = [
        [algos.cosine_score, "Cosine Similarity"]
        ,[algos.euclidean_score, "Euclidean Distance"]
        ,[algos.manhattan_score, "Manhattan Distance"]
        ,[algos.minkowski3_score, "Minkowski Distance (order 3)"]
        ,[algos.minkowski4_score, "Minkowski Distance (order 4)"]
        ,[algos.jaccard_score, "Jaccard Distance"]
        ,[algos.mult_score, "Multiplied Ratios"]
        ,[algos.neighbors_rand, "Random Neighbors"]
        ,[algos.full_rand, "Random Songs"]
    ]
    keys = []
    for i in range(len(scores)):
        keys.append(scores[i][1])

    smoothnesses = []

    test_time = str(time.strftime("%y-%m-%d_%H%M"))

    if not os.path.exists('graph-results/{}'.format(test_time)):
        os.makedirs('graph-results/{}'.format(test_time))

    for i in range(len(scores)):
        print("\n\n{}".format(keys[i]))
        smoothies = []      # smoothness values for each collective playlist
        songlists = []      # the different playlists

        # for loop for testing different amounts of points in between
        for n_songs_reqd in range(teststart, teststart + testcount):
            songlist, smoothie = prodplay.makePlaylist(
                songdata, user_orig, user_dest, n_songs_reqd, model, scores[i][0]
            ) 
            
            smoothies.append(smoothie)
            print("{}: {}".format(n_songs_reqd, smoothie))
            songlists.append(songlist)

        coords = []
        for j in range(len(songlists)):
            v_points = []
            a_points = []

            for k in range(len(songlists[j])):
                v_points.append(songdata.loc[songlists[j][k]][1])
                a_points.append(songdata.loc[songlists[j][k]][0])

            wrapped_points = [v_points, a_points]

            # helper.graph('valence', 'arousal', wrapped_points, data_dim = 2, marker='.',
            #     file = 'graph-results/{}/result_{}_{}.png'.format(test_time, j + teststart, keys[i]),
            #     title = "Playlist Path ({} songs) using {}".format(j + teststart, keys[i])
            # ) 

            coords.append(wrapped_points)

        smoothest = np.argmin(smoothies)
        print(smoothest + teststart)

        helper.graph('valence', 'arousal', coords, data_dim = 2, line_count = len(coords),
            file = 'graph-results/{}/all_lines_{}.png'.format(test_time, keys[i]),
            title = "All Playlist Paths (from {} to {} songs) using {}".format(
                teststart, teststart + testcount - 1, keys[i])
        )
        helper.graph('valence', 'arousal', coords[smoothest], data_dim = 2, marker='.',
            file = 'graph-results/{}/smoothest_{}.png'.format(test_time, keys[i]),
            title = "Smoothest Playlist Path ({} songs) using {}".format(smoothest + teststart, keys[i])
        )
        
        # # PUT THE "smoothest path" ON A SPOTIFY PLAYLIST
        # track_ids = []
        # for i in range(len(songlists[smoothest])):
        #     track_ids.append(songdata.loc[songlists[smoothest][i]][2])
        # print(track_ids)

        # title = "Productivity Playlist Test " + test_time
        # helper.makeSpotifyList(sp, username, title, track_ids, False)

        smoothnesses.append(smoothies)

    helper.graph('length of playlist', 'smoothness of path', smoothnesses, 
        data_dim = 1, line_count = len(smoothnesses), legend = keys,
        file = 'graph-results/{}/comparison.png'.format(test_time),
        title = "Mean Squared Error of Playlists Generated by Different Distances"
    )
