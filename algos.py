import pandas as pd
import numpy as np
import random
import warnings

def manhattan_score(cand, current, destination, songs_left):
    return minkowski_score(cand, current, destination, songs_left, 1)

def euclidean_score(cand, current, destination, songs_left):
    return minkowski_score(cand, current, destination, songs_left, 2)

def minkowski3_score(cand, current, destination, songs_left):
    return minkowski_score(cand, current, destination, songs_left, 3)

def minkowski4_score(cand, current, destination, songs_left):
    return minkowski_score(cand, current, destination, songs_left, 4)

def minkowski_score(cand, current, destination, songs_left, order):
    step = [(destination[i] - current[i] + .0000001) / songs_left for i in range(2)]    

    score = np.power(
        np.power(cand[1] - (current[1] + step[1]), order) + 
        np.power(cand[0] - (current[0] + step[0]), order), 
        1/order
    )

    return score

def mult_score(cand, current, destination, songs_left):
    step = [(destination[i] - current[i] + .0000001) / songs_left for i in range(2)]
    distRatio = [(cand[i] - current[i]) / step[i] for i in range(2)]
    score = np.absolute(1 - distRatio[1]) * np.absolute(1 - distRatio[0])
    return score

def cosine_score(cand, current, destination, songs_left):
    #Vector A: the vector to the candidate
    dist = [cand[i] - current[i] for i in range(2)]
    dist_mag = np.sqrt(np.square(dist[1]) + np.square(dist[0]))

    #Vector B: the vector to the hypothetical target
    step = [(destination[i] - current[i] + .0000001) / songs_left for i in range(2)]
    step_mag = np.sqrt(np.square(step[1]) + np.square(step[0]))

    # cosine = A dot B / (mag(A) * mag(B))
    dot_product = dist[1] * step[1] + dist[0] * step[0]
    mag_product = step_mag * dist_mag

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cosine = dot_product / mag_product

    # cos = 1 means closest, cos = -1 means farthest, so 1 is "smoother"
    # find difference between 1 and this cosine value
    score = 1 - cosine
    return score

def jaccard_score(cand, current, destination, songs_left):
    step = [(destination[i] - current[i] + .0000001) / songs_left for i in range(2)]
    target = [current[i] + step[i] for i in range(2)]

    #compute the weighted Jaccard similarity
    min_sum = min(cand[1], target[1]) + min(cand[0], target[0])
    max_sum = max(cand[1], target[1]) + max(cand[0], target[0])
    jaccard = min_sum / max_sum
    
    #Jaccard distance = 1 - Jaccard Similarity
    score = 1 - jaccard
    return score

def neighbors_rand(candidates, origin, destination):
    cand = candidates[random.randrange(0, len(candidates))]
    smooth = smoothness_mse(cand, origin, destination)
    return cand, smooth

def full_rand(coords, pointlist, origin, destination):
    pointFound = False
    while (pointFound == False):
        cand = coords[random.randrange(0, len(coords))]
        pointFound == True

        for j in range(len(pointlist)):
            if (abs(cand[0] - pointlist[j][0]) < .0000001 and abs(cand[1] - pointlist[j][1]) < .0000001):
                pointFound == False     

    smooth = smoothness_mse(cand, origin, destination)
    return cand, smooth

def smoothness_mse(current, origin, destination):
    slope = (destination[1] - origin[1]) / (destination[0] - origin[0])
    return np.square(current[1] - (origin[1] + slope * (current[0] - origin[0])))