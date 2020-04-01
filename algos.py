import pandas as pd
import numpy as np
import random

def euclidean_score(songdata, num, current, destination, songs_left):
    return minkowski_score(songdata, num, current, destination, songs_left, 2)

def manhattan_score(songdata, num, current, destination, songs_left):
    return minkowski_score(songdata, num, current, destination, songs_left, 1)

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

def rand_score(songdata, candidates, destination, origin, n_songs_reqd, songs_so_far):
    num = candidates[random.randrange(0, len(candidates))]
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