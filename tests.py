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

def get_points(songdata, used, target = 1):
    dist = 0.00

    while (dist < (0.95 * target) or dist > (1.05 * target)):
        orig = list(songdata.index.values)[random.randint(0, len(songdata)-1)]
        dest = list(songdata.index.values)[random.randint(0, len(songdata)-1)]

        origPt = songdata.loc[orig]
        destPt = songdata.loc[dest]
        
        while (orig, dest) in used:
            orig = songdata.iloc[random.randint(0, len(songdata)-1)]
            dest = songdata.iloc[random.randint(0, len(songdata)-1)]

        dist = np.sqrt(np.square(destPt[1] - origPt[1]) + np.square(destPt[0] - origPt[0]))

    used.append((orig, dest))
    return orig, dest

def test_lengths(dataset): 
    user_orig       = 762954
    user_dest       = 1157536
    neighbors       = 10
    n_songs_reqd    = [i for i in range(2, int(input("Max. Playlist Length: ")))]
    avgDists        = []
    listLengths     = []

    origPoint = dataset.data_df.loc[user_orig]
    destPoint = dataset.data_df.loc[user_dest]
    print(origPoint)
    print(destPoint)

    if dataset.knn_model is None:
        dataset.make_knn()

    for n in n_songs_reqd:
        newsongs, newsmooth, newpoints = prodplay.makePlaylist(
            dataset,
            user_orig, user_dest, n
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
            np.around(origPoint[0], decimals=2), np.around(origPoint[1], decimals=2), 
            np.around(destPoint[0], decimals=2), np.around(destPoint[1], decimals=2), 
        )
    )

    helper.graph('target playlist length', 'average dist. between points', [n_songs_reqd, avgDists], data_dim = 2, marker='.',
        file="graph-results/{}/avgDists.png".format(test_time),
        title = "Average Inter-Point Distances (K={}) from ({},{}) to ({},{})".format(
            neighbors,
            np.around(origPoint[0], decimals=2), np.around(origPoint[1], decimals=2), 
            np.around(destPoint[0], decimals=2), np.around(destPoint[1], decimals=2), 
        )
    )

def test_neighbors(dataset):
    the_dist = int(input("Distance: "))
    num_tests = int(input("Number of tests: "))
    maxlength = int(input("Max length: "))
    minlength = 2
    # neighbor_counts = [i * int(np.sqrt(len(coords)) / 2) for i in range(3,8)]
    neighbor_counts = [i for i in range(5, 30, 2)]
    test_time = str(time.strftime("%y-%m-%d_%H%M"))
    helper.makeDir('graph-results/{}'.format(test_time))
    total_smoothnesses = [[],[]]
    used_points = []

    if dataset.knn_model is None:
        dataset.make_knn()

    for c in range(num_tests):
        smoothnesses = []
        helper.makeDir('graph-results/{}/{}'.format(test_time, c))
        user_orig, user_dest = get_points(dataset.data_df, used_points, the_dist)
        
        origPoint = dataset.data_df.loc[user_orig]
        destPoint = dataset.data_df.loc[user_dest]
        print(origPoint)
        print(destPoint)

        for i in range(len(neighbor_counts)):
            helper.makeDir('graph-results/{}/{}/{}'.format(test_time, c, neighbor_counts[i]))
            print("\n\n{} Neighbors".format(neighbor_counts[i]))
            smoothies = [[],[]]         # smoothness values for each collective playlist
            songlists = []              # the different playlists
            pointlists = []

            # for loop for testing different amounts of points in between
            for n_songs_reqd in range(minlength, minlength + maxlength):
                songlist, smoothie, pointlist = prodplay.makePlaylist(
                    dataset, user_orig, user_dest, n_songs_reqd, neighbors=neighbor_counts[i]
                ) 

                print("{}: {}".format(len(songlist), smoothie))
                pointlist = np.transpose(pointlist)
                smoothies[1].append(smoothie)
                smoothies[0].append(len(songlist))
                songlists.append(songlist)
                pointlists.append(pointlist)

                # helper.graph('valence', 'arousal', pointlist, data_dim = 2, marker='.',
                #     file = 'graph-results/{}/{}/{}/{}.png'.format(test_time, c, neighbor_counts[i], len(songlist)),
                #     title = "Path ({} songs, K={}) from ({}, {}) to ({}, {})".format(
                #         len(songlist), neighbor_counts[i],
                #         np.around(origPoint[0], decimals=2), np.around(origPoint[1], decimals=2), 
                #         np.around(destPoint[0], decimals=2), np.around(destPoint[1], decimals=2), 
                #     )
                # )

            smoothest = np.argmin(smoothies[1])
            print(smoothest + minlength)

            helper.graph('valence', 'arousal', pointlists, data_dim = 2, line_count = len(pointlists),
                file = 'graph-results/{}/{}/{}/all.png'.format(test_time, c, neighbor_counts[i]),
                title = "Paths ({} to {} songs, K={}) from ({}, {}) to ({}, {})".format(
                    minlength, minlength + maxlength - 1, neighbor_counts[i],  
                    np.around(origPoint[0], decimals=2), np.around(origPoint[1], decimals=2), 
                    np.around(destPoint[0], decimals=2), np.around(destPoint[1], decimals=2)
                )
            )
            helper.graph('valence', 'arousal', pointlists[smoothest], data_dim = 2, marker='.',
                file = 'graph-results/{}/{}/{}/smoothest.png'.format(test_time, c, neighbor_counts[i]),
                title = "Smoothest Playlist Path ({} songs) using {} Neighbors from ({}, {}) to ({}, {})".format(
                    smoothest + minlength, neighbor_counts[i],
                    np.around(origPoint[0], decimals=2), np.around(origPoint[1], decimals=2), 
                    np.around(destPoint[0], decimals=2), np.around(destPoint[1], decimals=2)
                )
            )

            # title = "Productivity Playlist Test " + test_time
            # helper.makeSpotifyList(sp, username, title, track_ids, False)

            smoothnesses.append(smoothies)

        helper.graph('length of playlist', 'smoothness of path', smoothnesses, 
            data_dim = 2, line_count = len(smoothnesses), legend = neighbor_counts,
            file = 'graph-results/{}/{}/comparison.png'.format(test_time, c),
            title = "MSE Neighbor Comparison from ({}, {}) to ({}, {})".format(
                np.around(origPoint[0], decimals=2), np.around(origPoint[1], decimals=2), 
                np.around(destPoint[0], decimals=2), np.around(destPoint[1], decimals=2), 
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

def test_dists(dataset):
    # the_dist = int(input("Distance: "))
    maxlength = int(input("Max length: "))
    minlength = 2
    test_time = str(time.strftime("%y-%m-%d_%H%M"))
    helper.makeDir('graph-results/{}'.format(test_time))
    total_smoothnesses = [[],[]]
    used_points = []
    # user_orig, user_dest = get_points(dataset.data_df, used_points, the_dist)
    user_orig, user_dest = 3135555, 3135561
    
    origPoint = dataset.data_df.loc[user_orig]
    destPoint = dataset.data_df.loc[user_dest]
    print(origPoint)
    print(destPoint)

    if dataset.knn_model is None:
        dataset.make_knn()

    smoothnesses = []
    
    scores = [
        [algos.cosine_score, "Cosine Similarity"]
        ,[algos.euclidean_score, "Euclidean Distance"]
        # ,[algos.manhattan_score, "Manhattan Distance"]
        ,[algos.minkowski3_score, "Minkowski Distance (order 3)"]
        ,[algos.jaccard_score, "Jaccard Distance"]
        # ,[algos.mult_score, "Multiplied Ratios"]
        # ,[algos.neighbors_rand, "Random Neighbors"]
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
                dataset,
                user_orig, user_dest, n_songs_reqd, 
                score = scores[0][i]
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
            #         np.around(origPoint[0], decimals=2), np.around(origPoint[1], decimals=2), 
            #         np.around(destPoint[0], decimals=2), np.around(destPoint[1], decimals=2), 
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
                # np.around(origPoint[0], decimals=2), np.around(origPoint[1], decimals=2), 
                # np.around(destPoint[0], decimals=2), np.around(destPoint[1], decimals=2)
            )
        )
        helper.graph('valence', 'arousal', pointlists[smoothest], data_dim = 2, marker='.',
            file = 'graph-results/{}/{}/smoothest.png'.format(test_time, scores[1][i]),
            title = "Smoothest Playlist Path ({} songs) using {} from ({}, {}) to ({}, {})".format(
                smoothest + minlength, scores[1][i],
                np.around(origPoint[0], decimals=2), np.around(origPoint[1], decimals=2), 
                np.around(destPoint[0], decimals=2), np.around(destPoint[1], decimals=2)
            )
        )

    csv_data = {}
    for i in range(len(smoothnesses)):
        csv_data[scores[1][i]] = smoothnesses[i][1]

    helper.graph('length of playlist', 'smoothness of path', smoothnesses, 
        data_dim = 2, line_count = len(smoothnesses), legend = scores[1],
        file = 'graph-results/{}/comparison.png'.format(test_time),
        title = "MSE of Playlists by Distance"
    )

    # csv_df = pd.DataFrame(csv_data)
    # csv_df.to_csv("test_dist_output.csv".format(test_time))