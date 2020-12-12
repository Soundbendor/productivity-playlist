import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import time
import pprint

import algos
import helper
import prodplay

def get_points(songpoints, used, target = 1):
    dist = 0.00

    while (dist < (0.95 * target) or dist > (1.05 * target)):
        orig = songpoints.keys()[random.randint(0, len(songpoints)-1)]
        while orig in used:
            orig = songpoints.keys()[random.randint(0, len(songpoints)-1)]
        
        dest = songpoints.keys()[random.randint(0, len(songpoints)-1)]
        while dest in used:
            dest = songpoints.keys()[random.randint(0, len(songpoints)-1)]

        origArr = helper.string2arrPoint(orig)
        destArr = helper.string2arrPoint(dest)
        dist = np.sqrt(np.square(destArr[1] - origArr[1]) + np.square(destArr[0] - origArr[0]))

    used.append(orig)
    used.append(dest)

    origPt = songpoints[orig][random.randint(0, len(songpoints[orig])-1)]
    destPt = songpoints[dest][random.randint(0, len(songpoints[dest])-1)]
   
    print("\nEuclidean distance: {}".format(dist))
    print(origPt, destPt)
    return origPt, destPt

def test_lengths(model, songdata, songpoints, coords): 
    user_orig       = 762954
    user_dest       = 1157536
    neighbors       = 10
    n_songs_reqd    = [i for i in range(2, int(input("Max. Playlist Length: ")))]
    avgDists        = []
    listLengths     = []

    print(songdata.loc[user_orig])
    print(songdata.loc[user_dest])

    for n in n_songs_reqd:
        newsongs, newsmooth, newpoints = prodplay.makePlaylist(
            songdata, songpoints, coords, 
            user_orig, user_dest, n, 
            model
        )

        minDist = 20
        avgDist = 0
        for i in range(1, len(newpoints)):
            score = np.power(
                np.power(newpoints[i][0] - newpoints[i-1][0], 2) + 
                np.power(newpoints[i][1] - newpoints[i-1][1], 2), 
                1/2
            )
            minDist = min(minDist, score)
            avgDist = avgDist + score
        
        avgDist = avgDist / (n-1)

        listLengths.append(len(newsongs))
        avgDists.append(avgDist)

    pprint.pprint(newsongs)
    pprint.pprint(newsmooth)
    newpoints = np.transpose(newpoints)
    pprint.pprint(newpoints)

    test_time = str(time.strftime("%y-%m-%d_%H%M"))
    helper.makeDir('graph-results/{}'.format(test_time))

    helper.graph('target playlist length', 'actual playlist length', [n_songs_reqd, listLengths], data_dim = 2, marker='.',
        file="graph-results/{}/listLengths.png".format(test_time),
        title = "Playlist Length Comparison (K={}) from ({},{}) to ({},{})".format(
            neighbors,
            np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
            np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2), 
        )
    )

    helper.graph('target playlist length', 'average dist. between points', [n_songs_reqd, avgDists], data_dim = 2, marker='.',
        file="graph-results/{}/avgDists.png".format(test_time),
        title = "Average Inter-Point Distances (K={}) from ({},{}) to ({},{})".format(
            neighbors,
            np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
            np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2), 
        )
    )

# TODO: make tests of neighbors & distance vs. max playlist length
def test_neighbors(model, songdata, songpoints, coords):
    the_dist = int(input("Distance: "))
    num_tests = int(input("Number of tests: "))
    maxlength = int(input("Max length: "))
    minlength = 2
    # neighbor_counts = [i * int(np.sqrt(len(coords)) / 2) for i in range(3,8)]
    neighbor_counts = [i for i in range(5, 30, 2)]
    test_time = str(time.strftime("%y-%m-%d_%H%M"))
    helper.makeDir('graph-results/{}'.format(test_time))
    total_smoothnesses = [[],[]]
    used_points = ["-1, -1"]

    for c in range(num_tests):
        smoothnesses = []
        helper.makeDir('graph-results/{}/{}'.format(test_time, c))
        user_orig, user_dest = get_points(songpoints, used_points, the_dist)
        print(songdata.loc[user_orig])
        print(songdata.loc[user_dest])

        for i in range(len(neighbor_counts)):
            helper.makeDir('graph-results/{}/{}/{}'.format(test_time, c, neighbor_counts[i]))
            print("\n\n{} Neighbors".format(neighbor_counts[i]))
            smoothies = [[],[]]         # smoothness values for each collective playlist
            songlists = []              # the different playlists
            pointlists = []

            # for loop for testing different amounts of points in between
            for n_songs_reqd in range(minlength, minlength + maxlength):
                songlist, smoothie, pointlist = prodplay.makePlaylist(
                    songdata, songpoints, coords, 
                    user_orig, user_dest, n_songs_reqd, 
                    model, neighbors=neighbor_counts[i]
                ) 

                print("{}: {}".format(len(songlist), smoothie))
                pointlist = np.transpose(pointlist)
                smoothies[1].append(smoothie)
                smoothies[0].append(len(songlist))
                songlists.append(songlist)
                pointlists.append(pointlist)

                helper.graph('valence', 'arousal', pointlist, data_dim = 2, marker='.',
                    file = 'graph-results/{}/{}/{}/{}.png'.format(test_time, c, neighbor_counts[i], len(songlist)),
                    title = "Path ({} songs, K={}) from ({}, {}) to ({}, {})".format(
                        len(songlist), neighbor_counts[i],
                        np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
                        np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2), 
                    )
                )

            smoothest = np.argmin(smoothies[1])
            print(smoothest + minlength)

            helper.graph('valence', 'arousal', pointlists, data_dim = 2, line_count = len(pointlists),
                file = 'graph-results/{}/{}/{}/all.png'.format(test_time, c, neighbor_counts[i]),
                title = "Paths ({} to {} songs, K={}) from ({}, {}) to ({}, {})".format(
                    minlength, minlength + maxlength - 1, neighbor_counts[i],  
                    np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
                    np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2)
                )
            )
            helper.graph('valence', 'arousal', pointlists[smoothest], data_dim = 2, marker='.',
                file = 'graph-results/{}/{}/{}/smoothest.png'.format(test_time, c, neighbor_counts[i]),
                title = "Smoothest Playlist Path ({} songs) using {} Neighbors from ({}, {}) to ({}, {})".format(
                    smoothest + minlength, neighbor_counts[i],
                    np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
                    np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2)
                )
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
            data_dim = 2, line_count = len(smoothnesses), legend = neighbor_counts,
            file = 'graph-results/{}/{}/comparison.png'.format(test_time, c),
            title = "MSE Neighbor Comparison from ({}, {}) to ({}, {})".format(
                np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
                np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2), 
            )
        )

        if (total_smoothnesses != [[],[]]):
            for i in range(len(smoothnesses[1])):
                for j in range(len(smoothnesses[1][i])):
                    total_smoothnesses[1][i][j] += smoothnesses[1][i][j]
        else:
            total_smoothnesses = smoothnesses
        
    for i in range(len(total_smoothnesses[1])):
        for j in range(len(total_smoothnesses[1][i])):
            total_smoothnesses[1][i][j] /= num_tests

    helper.graph('length of playlist', 'smoothness of path', total_smoothnesses, 
        data_dim = 2, line_count = len(total_smoothnesses), legend = neighbor_counts,
        file = 'graph-results/{}/average.png'.format(test_time),
        title = "MSE Neighbor Comparisons Averaged over the Past {} Runs".format(num_tests)
    )

def test_dists(model, songdata, songpoints, coords):
    the_dist = int(input("Distance: "))
    maxlength = int(input("Max length: "))
    minlength = 2
    test_time = str(time.strftime("%y-%m-%d_%H%M"))
    helper.makeDir('graph-results/{}'.format(test_time))
    total_smoothnesses = [[],[]]
    used_points = ["-1, -1"]
    # user_orig, user_dest = get_points(songpoints, used_points, the_dist)
    user_orig       = 762954
    user_dest       = 1157536
    smoothnesses = []
    
    scores = [
        [algos.cosine_score, "Cosine Similarity"]
        ,[algos.euclidean_score, "Euclidean Distance"]
        ,[algos.manhattan_score, "Manhattan Distance"]
        ,[algos.minkowski3_score, "Minkowski Distance (order 3)"]
        ,[algos.jaccard_score, "Jaccard Distance"]
        ,[algos.mult_score, "Multiplied Ratios"]
        ,[algos.neighbors_rand, "Random Neighbors"]
    ]
    scores = np.transpose(scores)

    for i in range(len(scores[0])):
        helper.makeDir('graph-results/{}/{}'.format(test_time, scores[1][i]))
        print("\n\n{}".format(scores[1][i]))
        smoothies = [[],[]]     
        songlists = []
        pointlists = []

        # for loop for testing different amounts of points in between
        for n_songs_reqd in range(minlength, minlength + maxlength):
            songlist, smoothie, pointlist = prodplay.makePlaylist(
                songdata, songpoints, coords, 
                user_orig, user_dest, n_songs_reqd, 
                model, score = scores[0][i],
                neighbors = 10
            ) 

            print("{}: {}".format(len(songlist), smoothie))
            pointlist = np.transpose(pointlist)
            smoothies[1].append(smoothie)
            smoothies[0].append(n_songs_reqd)
            songlists.append(songlist)
            pointlists.append(pointlist)

            # helper.graph('valence', 'arousal', pointlist, data_dim = 2, marker='.',
            #     file = 'graph-results/{}/{}/{}.png'.format(test_time, scores[1][i], len(songlist)),
            #     title = "Playlist Path ({} songs using {}) from ({}, {}) to ({}, {})".format(
            #         len(songlist), scores[1][i],
            #         np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
            #         np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2), 
            #     )
            # ) 

        smoothest = np.argmin(smoothies[1])
        print(smoothest + minlength)
        smoothnesses.append(smoothies)

        helper.graph('valence', 'arousal', pointlists, data_dim = 2, line_count = len(pointlists),
            file = 'graph-results/{}/{}/all.png'.format(test_time, scores[1][i]),
            title = "Playlists Generated by {}".format(
                # minlength, minlength + maxlength - 1, 
                scores[1][i],  
                # np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
                # np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2)
            )
        )
        # helper.graph('valence', 'arousal', pointlists[smoothest], data_dim = 2, marker='.',
        #     file = 'graph-results/{}/{}/smoothest.png'.format(test_time, scores[1][i]),
        #     title = "Smoothest Playlist Path ({} songs) using {} from ({}, {}) to ({}, {})".format(
        #         smoothest + minlength, scores[1][i],
        #         np.around(songdata.loc[user_orig][0], decimals=2), np.around(songdata.loc[user_orig][1], decimals=2), 
        #         np.around(songdata.loc[user_dest][0], decimals=2), np.around(songdata.loc[user_dest][1], decimals=2)
        #     )
        # )
        
    # # PUT THE "smoothest path" ON A SPOTIFY PLAYLIST
        # track_ids = []
        # for i in range(len(songlists[smoothest])):
        #     track_ids.append(songdata.loc[songlists[smoothest][i]][2])
        # print(track_ids)

        # title = "Productivity Playlist Test " + test_time
        # helper.makeSpotifyList(sp, username, title, track_ids, False)

    helper.graph('length of playlist', 'smoothness of path', smoothnesses, 
        data_dim = 2, line_count = len(smoothnesses), legend = scores[1],
        file = 'graph-results/{}/comparison.png'.format(test_time),
        title = "MSE of Playlists by Distance"
    )