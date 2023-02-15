import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import algos
import helper
import pprint
import warnings
from songdataset import SongDataset

def filterCandidates(coords, pointlist, candidate_indices, candidate_dists, verbose = 0):
    dicts = []

    for x in range(len(candidate_indices)):
        i = candidate_indices[x]
        d = candidate_dists[x]

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
            
            dicts.append({"dist": d, "point": point})

    dicts = sorted(dicts, key=lambda d: d['dist'])
    filtered = [o["point"] for o in dicts]
    return np.array(filtered)

def getCandidates(dataset, pointlist, current, destination, n_songs_reqd, neighbors = 7, radius = 0.1, mode = "k", verbose = 0):
    distance = [destination[i] - current[i] + .0000001 for i in range(len(current))]
    remaining = n_songs_reqd - len(pointlist)
    target = [[current[i] + (distance[i]/remaining) for i in range(len(current))]]

    dists, nearest = dataset.knn_model.kneighbors(target, n_neighbors=3*neighbors, return_distance=True)
    # if mode == "radius":
    #     dists, nearest = dataset.knn_model.radius_neighbors(target, radius=radius, return_distance=True)
    # else:
    #     dists, nearest = dataset.knn_model.kneighbors(target, n_neighbors=3*neighbors, return_distance=True)
    
    candidate_indices = np.array(nearest)[0]
    candidate_dists = np.array(dists)[0]

    if verbose >= 1: print("Received {} candidates from KNN model using mode {}".format(len(candidate_indices), mode))
    if verbose >= 2:
        print("Index \t Dist (for candidates)")
        for i in range(len(candidate_indices)):
            print(candidate_indices[i], "\t", candidate_dists[i])

    candidates = filterCandidates(dataset.unique_points, pointlist, candidate_indices, candidate_dists)
    if verbose >= 2: print()
    if verbose >= 1: print("Filtering yields us {} candidates".format(len(candidates)))
    if verbose >= 2: print(candidates)

    if neighbors < len(candidates):
        candidates = candidates[0:neighbors]
    if verbose >= 1: print("Trimming to K={} yields {} candidates".format(neighbors, len(candidates)))

    return candidates

def chooseCandidate(candidates, current, origin, destination, nSongsReqd, pointLen, score, verbose = 0):
    songsLeft   = nSongsReqd - pointLen
    minScore    = None
    minSong     = None
    # minSmooth   = None

    if verbose >= 2: print("\nChoosing candidates!")
    for song, feats in candidates.iterrows():
        distScore = score(feats.tolist(), current, destination, songsLeft)
        # smoothness = algos.smoothness_mse(feats.tolist(), origin, destination, nSongsReqd, songsLeft)
        if verbose >= 2: print(" - score for {}\t is {}, minSong = {}, minScore = {}".format(song, np.around(distScore, decimals=8), minSong, minScore))

        if minScore is None or distScore < minScore:
            minScore   = distScore
            minSong    = song
            # minSmooth  = smoothness

    if verbose >= 2: print()
    if verbose >= 1: print("Winner: {}, score: {}".format(minSong, np.around(minScore, decimals=8)))
    return minSong

def makePlaylistDF(dataset, songs, points, feats, steps):
    length = len(songs)
    ids = []
    obj = {
        "id-deezer": [],
        "artist": [],
        "title": [],
        "id-spotify": [],
        "valence": [],
        "arousal": [],
        # "smoothness": [],
        "step": [],
    }

    for i in range(length):
        obj["id-deezer"].append(int(songs[i]))
        obj["artist"].append(dataset.full_df.loc[int(songs[i])]['artist_name'])
        obj["title"].append(dataset.full_df.loc[int(songs[i])]['track_name'])
        obj["id-spotify"].append(dataset.get_spid(int(songs[i])))
        obj["valence"].append(np.around(points[i][0], decimals=8))
        obj["arousal"].append(np.around(points[i][1], decimals=8))
        # obj["smoothness"].append(np.around(smooths[i], decimals=8))
        obj["step"].append(np.around(steps[i], decimals=8))

    df = pd.DataFrame(obj)
    return df


def makePlaylist(dataset, origin, destination, n_songs_reqd, score = algos.cosine_score, neighbors = 7, radius = 0.1, mode = "k", verbose = 0):

    if verbose >= 2: print("\n")
    if verbose >= 1: print("\n\nMAKING PLAYLIST")
    if verbose >= 2: print()

    if dataset.knn_model is None:
        dataset.make_knn()

    # n_songs_reqd -= 1
    songlist = np.empty(0)
    # smoothlist = np.empty(0)
    steplist = np.empty(0)
    featlist = []
    pointlist = []

    origPoint = dataset.get_point(origin).tolist()
    destPoint = dataset.get_point(destination).tolist()
    origFeats = dataset.get_feats(origin, "tail").tolist()
    destFeats = dataset.get_feats(destination, "head").tolist()

    songlist = np.append(songlist, origin)
    # smoothlist = np.append(smoothlist, 0)
    steplist = np.append(steplist, 0)
    pointlist.append(origPoint)
    featlist.append(origFeats)
    
    current = origin
    currPoint = origPoint
    currFeats = origFeats

    # Get middle points in the playlist.
    while ((len(pointlist) < n_songs_reqd - 1)):

        if verbose >= 1: print("\n{}) current song = {} ({},{})".format(len(songlist), current, np.around(currPoint[0], decimals=2), np.around(currPoint[1], decimals=2)))
        if verbose >= 2: print()

        # Step 1: Get candidate points from KNN in Valence-Arousal space.
        candPoints = getCandidates(
            dataset, pointlist, currPoint, destPoint, n_songs_reqd, neighbors, radius, mode, verbose
        )

        # Step 2: get candidate features.
        
        ## - Get all candidate IDs from points.
        candIDs = []
        for point in candPoints:
            candIDs.extend(dataset.get_all_songs(point))
        
        # Prevent early stoppage.
        if destination in candIDs:
            candIDs.remove(destination)
        
        if verbose >= 2: print()
        if verbose >= 1: print("Found {} songs from candidate points".format(len(candIDs)))

        ## - Get features from candidate IDs.
        ## - We want to get the heads of the next song choices to transition right for them.
        candFeatsDF = dataset.get_feats(candIDs, "head")
        if verbose >= 2: print("Found features from candidate songs!\n")
        if verbose >= 2: print()

        # Step 3: Use a distance score to evaluate candidate features.
        ## - Return song ID based on distance score from features.
        nextSong = chooseCandidate(
            candFeatsDF, currFeats, origFeats, destFeats, n_songs_reqd, len(pointlist), score, verbose
        )
        nextPoint = dataset.get_point(nextSong).tolist()
        nextFeats = dataset.get_feats(nextSong, "tail").tolist()

        # smoothlist = np.append(smoothlist, nextSmooth)
        steplist = np.append(steplist, np.linalg.norm(
            np.array(nextPoint) - np.array(currPoint))
        )
        songlist = np.append(songlist, nextSong)
        pointlist.append(nextPoint)

        current = nextSong
        currPoint = nextPoint
        currFeats = nextFeats
    
    # Put last point in the playlist.
    if (current != destination):
        pointlist.append(destPoint)
        songlist = np.append(songlist, destination)
        featlist.append(destFeats)
        steplist = np.append(steplist, np.linalg.norm(
            np.array(destPoint) - np.array(currPoint)
        ))

    # smoothness = np.mean(smoothlist)
    # evenness = np.var(steplist)
    if (verbose >= 1): print("\nPLAYLIST DONE\n\n")
    playlistDF = makePlaylistDF(dataset, songlist, np.array(pointlist), np.array(featlist), steplist)
    return playlistDF