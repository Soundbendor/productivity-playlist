import pandas as pd
import numpy as np
import random

def manhattan_score(songdata, num, current, destination, songs_left):
    return minkowski_score(songdata, num, current, destination, songs_left, 1)

def euclidean_score(songdata, num, current, destination, songs_left):
    return minkowski_score(songdata, num, current, destination, songs_left, 2)

def minkowski3_score(songdata, num, current, destination, songs_left):
    return minkowski_score(songdata, num, current, destination, songs_left, 3)

def minkowski4_score(songdata, num, current, destination, songs_left):
    return minkowski_score(songdata, num, current, destination, songs_left, 4)

def minkowski_score(songdata, num, current, destination, songs_left, order):
    current_a = songdata.loc[current][1]
    current_v = songdata.loc[current][0]
    destination_a = songdata.loc[destination][1]
    destination_v = songdata.loc[destination][0]
    step_a = (destination_a - current_a + .001) / songs_left
    step_v = (destination_v - current_v + .001) / songs_left

    score = np.power(
        np.power(songdata.iloc[num][1] - (current_a + step_a), order) + 
        np.power(songdata.iloc[num][0] - (current_v + step_v), order), 
        1/order
    )

    return score

def mult_score(songdata, num, current, destination, songs_left):
    current_a = songdata.loc[current][1]
    current_v = songdata.loc[current][0]
    destination_a = songdata.loc[destination][1]
    destination_v = songdata.loc[destination][0]
    step_a = (destination_a - current_a + .001) / songs_left
    step_v = (destination_v - current_v + .001) / songs_left
    
    dist_a_ratio = (songdata.iloc[num][1] - current_a) / step_a
    dist_v_ratio = (songdata.iloc[num][0] - current_v) / step_v
    
    score = np.absolute(1 - dist_a_ratio) * np.absolute(1 - dist_v_ratio)
    return score

def cosine_score(songdata, num, current, destination, songs_left):
    current_a = songdata.loc[current][1]
    current_v = songdata.loc[current][0]
    destination_a = songdata.loc[destination][1]
    destination_v = songdata.loc[destination][0]
    step_a = (destination_a - current_a + .000000001) / songs_left
    step_v = (destination_v - current_v + .000000001) / songs_left
    
    #Vector A: the vector to the candidate
    dist_a = songdata.iloc[num][1] - current_a
    dist_v = songdata.iloc[num][0] - current_v
    dist_mag = np.sqrt(np.square(dist_a) + np.square(dist_v))

    #Vector B: the vector to the hypothetical target
    step_a = (destination_a - current_a + .000000001) / songs_left
    step_v = (destination_v - current_v + .000000001) / songs_left
    step_mag = np.sqrt(np.square(step_a) + np.square(step_v))

    # cosine = A dot B / (mag(A) * mag(B))
    dot_product = dist_a * step_a + dist_v * step_v
    mag_product = step_mag * dist_mag
    cosine = dot_product / mag_product

    # cos = 1 means closest, cos = -1 means farthest, so 1 is "smoother"
    # find difference between 1 and this cosine value
    score = 1 - cosine
    return score

def jaccard_score(songdata, num, current, destination, songs_left):
    current_a = songdata.loc[current][1]
    current_v = songdata.loc[current][0]
    destination_a = songdata.loc[destination][1]
    destination_v = songdata.loc[destination][0]
    step_a = (destination_a - current_a + .0000001) / songs_left
    step_v = (destination_v - current_v + .0000001) / songs_left
    target_a = current_a + step_a
    target_v = current_v + step_v

    #compute the weighted Jaccard similarity
    min_sum = min(songdata.iloc[num][1], target_a) + min(songdata.iloc[num][0], target_v)
    max_sum = max(songdata.iloc[num][1], target_a) + max(songdata.iloc[num][0], target_v)
    jaccard = min_sum / max_sum
    
    #Jaccard distance = 1 - Jaccard Similarity
    score = 1 - jaccard
    return score

def neighbors_rand(songdata, candidates, origin, destination):
    num = candidates[random.randrange(0, len(candidates))]
    smooth = smoothness_mse(songdata, origin, destination, num)
    return num, smooth

def full_rand(songdata, origin, destination):
    num = random.randrange(0, len(list(songdata.index.values)))
    smooth = smoothness_mse(songdata, origin, destination, num)
    return num, smooth

def smoothness_mse(songdata, origin, destination, num):
    origin_a = songdata.loc[origin][1]
    origin_v = songdata.loc[origin][0]
    destination_a = songdata.loc[destination][1]
    destination_v = songdata.loc[destination][0]
    current_a = songdata.iloc[num][1]
    current_v = songdata.iloc[num][0]

    slope = (destination_a - origin_a) / (destination_v - origin_v)
    return np.square(current_a - (origin_a + slope * (current_v - origin_v)))