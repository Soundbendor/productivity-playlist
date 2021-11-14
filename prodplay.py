import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import algos
import helper
import pprint
import warnings

def filter_candiates(coords, pointlist, candidate_indices):
    filtered = []

    for i in candidate_indices:
        unique = True
        j = 0
        while unique and j < len(pointlist):
            bad = True
            k = 0
            while k < len(coords[i]) and bad:
                if (abs(coords[i][k] - pointlist[j][k]) > .000000000001):
                    bad = False
                k = k + 1
            if bad:
                unique = False
            j = j + 1

        if (unique == True):
            filtered.append(coords[i].tolist())

    return np.array(filtered)

def get_candidates(coords, pointlist, current, destination, n_songs_reqd, model, neighbors = 3):
    distance = [destination[i] - current[i] + .0000001 for i in range(len(current))]
    remaining = n_songs_reqd - len(pointlist) + 1
    target = [[current[i] + (distance[i]/remaining) for i in range(len(current))]]

    nearest = model.kneighbors(target, n_neighbors=(n_songs_reqd * neighbors))
    candidate_indices = np.array(nearest[1])[0]
    candidates = filter_candiates(coords, pointlist, candidate_indices)
    return candidates

def choose_candidate(candidates, current, origin, destination, songs_left, score):
    candScores = []
    candSmooth = []

    for cand in candidates:
        candScores.append(score(cand, current, destination, songs_left))
        candSmooth.append(algos.smoothness_mse(cand, origin, destination))

    choice = np.argmin(candScores)
    return candidates[choice].tolist(), candSmooth[choice]

def makePlaylist(songdata, coords, origin, destination, n_songs_reqd, model, score = algos.cosine_score, neighbors = 7, si = 0, songobj = None):
    n_songs_reqd -= 1
    smoothlist = np.empty(0)
    songlist = np.empty(0)
    pointlist = []

    origPoint = songdata.loc[origin][si:].tolist()
    destPoint = songdata.loc[destination][si:].tolist()

    songlist = np.append(songlist, origin)
    pointlist.append(origPoint)
    currPoint = origPoint

    while ((len(pointlist) < n_songs_reqd) and currPoint != destPoint):
        if (score == algos.full_rand):
            nextPoint, nextSmooth = score(coords, pointlist, origPoint, destPoint)
        else:
            candidates = get_candidates(coords, pointlist, currPoint, destPoint, n_songs_reqd, model, neighbors)
            if (score == algos.neighbors_rand):
                nextPoint, nextSmooth = score(candidates, origPoint, destPoint)
            else:
                nextPoint, nextSmooth = choose_candidate(candidates, currPoint, origPoint, destPoint, n_songs_reqd - len(pointlist) + 1, score)

        if (nextPoint != destPoint):
            if songobj is None:
                i = 0
                found = False
                while i < len(songdata) and not found:
                    j = 0
                    good = True
                    while j < len(nextPoint) and good:
                        if (nextPoint[j] != songdata.iloc[i][j+si]):
                            good = False
                        j = j + 1
                    
                    if good:
                        found = True
                    else:
                        i = i + 1
                
                nextSong = songdata.index.values[i]
            else:
                nextString = helper.arr2stringPoint(nextPoint)
                nextSong = songobj[nextString][random.randint(0, len(songobj[nextString])-1)]
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
