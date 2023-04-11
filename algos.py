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
    step = [(destination[i] - current[i] + .0000001) / songs_left for i in range(len(destination))]    

    s = 0
    for i in range(len(destination)):
        s = s + np.power(abs(cand[i] - (current[i] + step[i])), order) 
    
    score = np.power(s, 1/order)
    return score

def mult_score(cand, current, destination, songs_left):
    step = [(destination[i] - current[i] + .0000001) / songs_left for i in range(len(destination))]
    distRatio = [(cand[i] - current[i]) / step[i] for i in range(len(destination))]
    
    prod = 1
    for i in range(len(destination)):
        prod = prod * (1 + np.absolute(1 - distRatio[i]))
    
    score = 1 - prod
    return score

def cosine_score(cand, current, destination, songs_left):
    #Vector A: the vector to the candidate
    dist = [cand[i] - current[i] for i in range(len(destination))]
    dist_sum = 0
    for i in range(len(destination)):
        dist_sum = dist_sum + np.square(dist[i])
    dist_mag = np.sqrt(dist_sum)

    #Vector B: the vector to the hypothetical target
    step = [(destination[i] - current[i] + .0000001) / songs_left for i in range(len(destination))]
    step_sum = 0
    for i in range(len(destination)):
        step_sum = step_sum + np.square(step[i])
    step_mag = np.sqrt(step_sum)

    # cosine = A dot B / (mag(A) * mag(B))
    dot_product = 0
    for i in range(len(destination)):
        dot_product = dot_product + (dist[i] * step[i])
    mag_product = step_mag * dist_mag

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cosine = dot_product / mag_product

    # cos = 1 means closest, cos = -1 means farthest, so 1 is "smoother"
    # find difference between 1 and this cosine value
    score = 1 - cosine
    return score

def jaccard_score(cand, current, destination, songs_left):
    step = [(destination[i] - current[i] + .0000001) / songs_left for i in range(len(destination))]
    target = [current[i] + step[i] for i in range(len(destination))]

    #compute the weighted Jaccard similarity
    min_sum, max_sum = 0, 0
    for i in range(len(destination)):
        min_sum = min_sum + min(cand[i], target[i])
        max_sum = max_sum + max(cand[i], target[i])
    jaccard = min_sum / max_sum
    
    #Jaccard distance = 1 - Jaccard Similarity
    score = 1 - jaccard
    return score

def neighbors_rand(cand, current, destination, songs_left):
    score = random.random()
    return score

# -------------------------------------------------------------------------------
# Commented because out of date.
# -------------------------------------------------------------------------------
# def full_rand(coords, pointlist, origin, destination):
#     pointFound = False
#     while (pointFound == False):
#         cand = coords[random.randrange(0, len(coords))]
#         pointFound == True

#         for j in range(len(pointlist)):
#             if (abs(cand[0] - pointlist[j][0]) < .0000001 and abs(cand[1] - pointlist[j][1]) < .0000001):
#                 pointFound == False     

#     smooth = smoothness_mse(cand, origin, destination)
#     return cand, smooth

def smoothness_mse(current, origin, destination, n_songs_reqd, songs_left):
    n = len(destination)
    step = [(destination[i] - origin[i] + .0000001) / n_songs_reqd for i in range(n)]
    target = [(destination[i] - songs_left * step[i]) for i in range(n)]
    squaresum = sum([np.square(current[i] - target[i]) for i in range(n)])
    return squaresum / n