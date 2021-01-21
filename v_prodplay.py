import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import v_algos as algos
import helper
import pprint
import warnings

def filter_candiates(coords, pointlist, candidates):
    filtered = []

    for i in candidates:
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
            filtered.append(coords[i].tolist())
    
    return np.array(filtered)

def get_candidates(coords, pointlist, current, destination, n_songs_reqd, model, neighbors = 7):
    distance = [destination[i] - current[i] + .0000001 for i in range(len(current))]
    remaining = n_songs_reqd - len(pointlist) + 1
    target = [[current[i] + (distance[i]/remaining) for i in range(len(current))]]

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

def makePlaylist(songdata, coords, origin, destination, n_songs_reqd, model, score = algos.cosine_score, neighbors = 7):
    smoothlist = np.empty(0)
    songlist = np.empty(0)
    pointlist = []

    origPoint = songdata.loc[origin][1:].tolist()
    destPoint = songdata.loc[destination][1:].tolist()

    songlist = np.append(songlist, origin)
    pointlist.append(origPoint)
    currPoint = origPoint

    while ((len(pointlist) < n_songs_reqd) and currPoint != destPoint):
        print(len(pointlist), end="\r")
        
        if (score == algos.full_rand):
            nextPoint, nextSmooth = score(coords, pointlist, origPoint, destPoint)
        else:
            candidates = get_candidates(coords, pointlist, currPoint, destPoint, n_songs_reqd, model, neighbors)
            # pprint.pprint(candidates)
            if (score == algos.neighbors_rand):
                nextPoint, nextSmooth = score(candidates, origPoint, destPoint)
            else:
                nextPoint, nextSmooth = choose_candidate(candidates, currPoint, origPoint, destPoint, n_songs_reqd - len(pointlist) + 1, score)

        if (nextPoint != destPoint):
            i = 0
            found = False
            while i < len(songdata) and not found:
                j = 0
                good = True
                while j < len(nextPoint) and good:
                    if (nextPoint[j] != songdata.iloc[i][j+1]):
                        good = False
                    j = j + 1
                
                if good:
                    found = True
                else:
                    i = i + 1
            
            nextSong = songdata.index.values[i]
        else:
            nextSong = destination

        smoothlist = np.append(smoothlist, nextSmooth)
        songlist = np.append(songlist, nextSong)
        pointlist.append(nextPoint)

        currPoint = nextPoint
    
    if (pointlist[len(pointlist) - 1] != destPoint):
        pointlist.append(destPoint)
        songlist = np.append(songlist, destination)

    smoothness = np.mean(smoothlist)
    return songlist, smoothness, np.array(pointlist)
