import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import algos
import helper

def filter_candiates(coords, pointlist, candidates):
    for i in candidates:
        unique = True
        for p in pointlist:
            if (coords[i] == p):
                unique == False
        if (unique == True):
            filtered.append(coords[i])
    
    np.array(filtered)

def get_candidates(coords, pointlist, current, destination, n_songs_reqd, model, neighbors = 7):
    distance = [destination[i] - current[i] + .0000001 for i in range(2)]
    remaining = n_songs_reqd - len(pointlist) + 1
    target = [current[i] + (distance[i]/remaining) for i in range(2)]

    candidates = []
    multiplier = 1
    while (candidates == []):
        candidates = np.array(model.kneighbors(target, n_neighbors=(neighbors*multiplier))[1])[0]
        candidates = filter_candiates(coords, pointlist, candidates)
        multiplier = multiplier + 1

    return candidates

def choose_candidate(songdata, candidates, current, origin, destination, songs_left, score):
    candScores = []
    candSmooth = []

    for cand in candidates:
        candScores.append(score(cand, current, destination, songs_left))
        candSmooth.append(algos.smoothness_mse(cand, origin, destination))

    choice = np.argmin(candScores)
    return candidates[choice], candSmooth[choice]

def makePlaylist(songdata, songpoints, coords, origin, destination, n_songs_reqd, model, score = algos.cosine_score, neighbors = 19):
    smoothlist = np.empty(0)
    songlist = np.empty(0)
    pointlist = np.empty(0)

    origPoint = [songdata.loc[origin][0], songdata.loc[origin][1]]
    destPoint = [songdata.loc[destination][0], songdata.loc[destination][1]]

    songlist.append(origin)
    pointlist.append(origPoint)
    currPoint = origPoint

    while ((helper.arr2stringPoint(currPoint) != helper.arr2stringPoint(destPoint)) & (len(pointlist) - 1 < n_songs_reqd)):
        
        if (score == algos.full_rand):
            nextPoint, nextSmooth = score(coords, pointlist, origin, destination)
        else:
            candidates = get_candidates(coords, pointlist, currPoint, destPoint, n_songs_reqd, model, neighbors)
            if (score == algos.neighbors_rand):
                nextPoint, nextSmooth = score(candidates, origin, destination)
            else:
                nextPoint, nextSmooth = choose_candidate(candidates, currPoint, origPoint, destPoint, n_songs_reqd - len(pointlist) + 1, score)
        
        
        nextString = helper.arr2stringPoint(nextPoint)
        if (nextPoint != destPoint):
            nextSong = songpoints[nextString][random.randint(0, len(songpoints[nextString])-1)]
        else:
            nextSong = destination

        smoothlist.append(nextSmooth)
        pointlist.append(nextPoint)
        songlist.append(nextSong)

        currPoint = nextPoint
    
    pointlist = pd.unique(np.append(pointlist, destPoint))
    songlist = pd.unique(np.append(songlist, destination))

    smoothness = np.mean(smoothlist)
    return songlist, smoothness, pointlist
