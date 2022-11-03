import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import algos
import helper
import pprint
import warnings
from songdataset import SongDataset

def filterCandidates(coords, pointlist, candidate_indices, verbose = 0):
    filtered = []

    for i in candidate_indices:
        unique = True
        j = 0
        while unique and j < len(pointlist):
            bad = True
            k = 0
            while k < len(coords[i]) and bad:
                if (abs(coords[i][k] - pointlist[j][k]) > .0000001):
                    bad = False
                k = k + 1
            if bad:
                unique = False
            j = j + 1

        if (unique == True):
            point = coords[i].tolist()
            if (verbose >= 2):
                print("Point {} is good!".format(point))
            filtered.append(point)

    return np.array(filtered)

def getCandidates(dataset, pointlist, current, destination, n_songs_reqd, neighbors = 7, radius = 0.1, mode = "radius", verbose = 0):
    distance = [destination[i] - current[i] + .0000001 for i in range(len(current))]
    remaining = n_songs_reqd - len(pointlist) + 1
    target = [[current[i] + (distance[i]/remaining) for i in range(len(current))]]

    nearest = None
    if mode == "radius":
        nearest = dataset.knn_model.radius_neighbors(target, radius=radius, return_distance=False)
    else:
        nearest = dataset.knn_model.kneighbors(target, n_neighbors=neighbors, return_distance=False)
    
    candidate_indices = np.array(nearest)[0]
    if verbose >= 1: print("Received {} candidates from Neighbor model using mode {}".format(len(candidate_indices), mode))

    candidates = filterCandidates(dataset.unique_points, pointlist, candidate_indices)
    if verbose >= 1: print("Filtering yields us {} candidates".format(len(candidates)))
    return candidates

def chooseCandidate(candidates, current, origin, destination, nSongsReqd, pointLen, score, verbose = 0):
    songsLeft   = nSongsReqd - pointLen + 1
    minScore    = None
    minSong     = None
    minSmooth   = None

    if verbose >= 2: print("\nChoosing candidates!")
    for song, feats in candidates.iterrows():
        distScore = score(feats.tolist(), current, destination, songsLeft)
        smoothness = algos.smoothness_mse(feats.tolist(), origin, destination, nSongsReqd, songsLeft)
        if verbose >= 2: print(" - score for {}\t is {} with smoothness {}".format(song, np.around(distScore, decimals=8), np.around(smoothness, decimals=8)))

        if minScore is None or distScore < minScore:
            minScore   = distScore
            minSong    = song
            minSmooth  = smoothness

    if verbose >= 2: print()
    if verbose >= 1: print("Winner: {}, score: {}, smooth: {}".format(minSong, np.around(minScore, decimals=8), np.around(minSmooth, decimals=8)))
    return minSong, minSmooth

def makePlaylistDF(dataset, songs, points, feats, smooths, steps):
    length = len(songs)
    ids = []
    obj = {
        "id-deezer": [],
        "artist": [],
        "title": [],
        "id-spotify": [],
        "valence": [],
        "arousal": [],
        "smoothness": [],
        "evenness": [],
    }

    

    for i in range(length):
        obj["id-deezer"].append(int(songs[i]))
        obj["artist"].append(dataset.full_df.loc[int(songs[i])]['artist_name'])
        obj["title"].append(dataset.full_df.loc[int(songs[i])]['track_name'])
        obj["id-spotify"].append(dataset.get_spid(int(songs[i])))
        obj["valence"].append(np.around(points[i][0], decimals=8))
        obj["arousal"].append(np.around(points[i][1], decimals=8))
        obj["smoothness"].append(np.around(smooths[i], decimals=8))
        obj["evenness"].append(np.around(steps[i], decimals=8))

    df = pd.DataFrame(obj)
    return df


def makePlaylist(dataset, origin, destination, n_songs_reqd, score = algos.cosine_score, neighbors = 7, radius = 0.1, verbose = 0):

    if verbose >= 2: print("\n")
    if verbose >= 1: print("\n\nMAKING PLAYLIST")
    if verbose >= 2: print()

    if dataset.knn_model is None:
        dataset.make_knn()

    n_songs_reqd -= 1
    songlist = np.empty(0)
    smoothlist = np.empty(0)
    steplist = np.empty(0)
    featlist = []
    pointlist = []

    origPoint = dataset.va_df.loc[origin].tolist()
    destPoint = dataset.va_df.loc[destination].tolist()
    origFeats = dataset.data_df.loc[origin].tolist()
    destFeats = dataset.data_df.loc[destination].tolist()

    songlist = np.append(songlist, origin)
    smoothlist = np.append(smoothlist, 0)
    steplist = np.append(steplist, 0)
    pointlist.append(origPoint)
    featlist.append(origFeats)
    
    current = origin
    currPoint = origPoint
    currFeats = origFeats

    while ((len(pointlist) <= n_songs_reqd) and current != destination):

        if verbose >= 1: print("\n{}) current song = {} ({},{})".format(len(songlist), current, np.around(currPoint[0], decimals=2), np.around(currPoint[1], decimals=2)))
        if verbose >= 2: print()

        # Step 1: Get candidate points from KNN in Valence-Arousal space.
        candPoints = getCandidates(
            dataset, pointlist, currPoint, destPoint, n_songs_reqd, neighbors, radius, "radius", verbose
        )

        # Step 2: get candidate features.
        
        ## - Get all candidate IDs from points.
        candIDs = []
        for point in candPoints:
            candIDs.extend(dataset.get_all_songs(point))
        
        if verbose >= 2: print()
        if verbose >= 1: print("Found {} songs from candidate points".format(len(candIDs)))

        ## - Get features from candidate IDs.
        candFeatsDF = dataset.data_df.loc[candIDs]
        if verbose >= 2: print("Found features from candidate songs!\n")
        if verbose >= 2: print()

        # Step 3: Use a distance score to evaluate candidate features.
        ## - Return song ID based on distance score from features.
        nextSong, nextSmooth = chooseCandidate(
            candFeatsDF, currFeats, origFeats, destFeats, n_songs_reqd, len(pointlist), score, verbose
        )
        nextPoint = dataset.va_df.loc[nextSong].tolist()
        nextFeats = dataset.data_df.loc[nextSong].tolist()

        smoothlist = np.append(smoothlist, nextSmooth)
        steplist = np.append(steplist, np.linalg.norm(
            np.array(nextPoint) - np.array(currPoint))
        )
        songlist = np.append(songlist, nextSong)
        pointlist.append(nextPoint)

        current = nextSong
        currPoint = nextPoint
        currFeats = nextFeats
    
    if (songlist[len(songlist) - 1] != destination):
        pointlist.append(destPoint)
        songlist = np.append(songlist, destination)

    # smoothness = np.mean(smoothlist)
    # evenness = np.var(steplist)
    if (verbose >= 1): print("\nPLAYLIST DONE\n\n")
    playlistDF = makePlaylistDF(dataset, songlist, np.array(pointlist), np.array(featlist), smoothlist, steplist)
    return playlistDF