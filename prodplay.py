import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import algos
import helper
import pprint

def filter_candiates(coords, pointlist, candidates):
    filtered = []

    for i in candidates:
        unique = True
        for j in range(len(pointlist)):
            # print("Checking {} vs {} -> {}".format(coords[i], pointlist[j], abs(coords[i][0] - pointlist[j][0]) < .0000001 and abs(coords[i][1] - pointlist[j][1]) < .0000001))
            if (abs(coords[i][0] - pointlist[j][0]) < .0000001 and abs(coords[i][1] - pointlist[j][1]) < .0000001):
                unique = False
        if (unique == True):
            filtered.append(coords[i].tolist())
    
    return np.array(filtered)

def get_candidates(coords, pointlist, current, destination, n_songs_reqd, model, neighbors = 7):
    distance = [destination[i] - current[i] + .0000001 for i in range(2)]
    remaining = n_songs_reqd - len(pointlist) + 1
    target = [[current[i] + (distance[i]/remaining) for i in range(2)]]

    # print("C: {}, D: {}, dist = {}, r: {}, T: {}".format(current, destination, distance, remaining, target))

    candidates = []
    multiplier = 1
    while (candidates == []):
        nearest = model.kneighbors(target, n_neighbors=neighbors)
        # pprint.pprint(nearest[0][0])
        candidates = np.array(nearest[1])[0]
        candidates = filter_candiates(coords, pointlist, candidates)
        multiplier = multiplier + 1

    return candidates

def choose_candidate(candidates, current, origin, destination, songs_left, score):
    candScores = []
    candSmooth = []

    for cand in candidates:
        candScores.append(score(cand, current, destination, songs_left))
        candSmooth.append(algos.smoothness_mse(cand, origin, destination))

    choice = np.argmin(candScores)
    return candidates[choice].tolist(), candSmooth[choice]

def makePlaylist(songdata, songpoints, coords, origin, destination, n_songs_reqd, model, score = algos.cosine_score, neighbors = 7):
    smoothlist = np.empty(0)
    songlist = np.empty(0)
    pointlist = []

    origPoint = [songdata.loc[origin][0], songdata.loc[origin][1]]
    destPoint = [songdata.loc[destination][0], songdata.loc[destination][1]]

    songlist = np.append(songlist, origin)
    pointlist.append(origPoint)
    currPoint = origPoint

    while ((len(pointlist) - 1 < n_songs_reqd) and (helper.arr2stringPoint(currPoint) != helper.arr2stringPoint(destPoint))):
        
        if (score == algos.full_rand):
            nextPoint, nextSmooth = score(coords, pointlist, origin, destination)
        else:
            candidates = get_candidates(coords, pointlist, currPoint, destPoint, n_songs_reqd, model, neighbors)
            # pprint.pprint(candidates)
            if (score == algos.neighbors_rand):
                nextPoint, nextSmooth = score(candidates, origin, destination)
            else:
                nextPoint, nextSmooth = choose_candidate(candidates, currPoint, origPoint, destPoint, n_songs_reqd - len(pointlist) + 1, score)

        nextString = helper.arr2stringPoint(nextPoint)
        if (helper.arr2stringPoint(nextPoint) != helper.arr2stringPoint(destPoint)):
            nextSong = songpoints[nextString][random.randint(0, len(songpoints[nextString])-1)]
        else:
            nextSong = destination

        smoothlist = np.append(smoothlist, nextSmooth)
        songlist = np.append(songlist, nextSong)
        pointlist.append(nextPoint)

        currPoint = nextPoint
    
    # pointlist.append(destPoint)
    # songlist = np.append(songlist, destination)

    smoothness = np.mean(smoothlist)
    return songlist, smoothness, np.array(pointlist)
