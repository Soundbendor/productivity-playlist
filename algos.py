import pandas as pd
import numpy as np

def euclid_dist(x1, y1, x2, y2):
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

def euclid_score(songdata, num, current, destination, songs_left):
    current_a = songdata.loc[current][1]
    current_v = songdata.loc[current][0]
    destination_a = songdata.loc[destination][1]
    destination_v = songdata.loc[destination][0]
    step_a = (destination_a - current_a + .001) / songs_left
    step_v = (destination_v - current_v + .001) / songs_left

    score = euclid_dist(
        current_v + step_v, current_a + step_a, 
        songdata.iloc[num][0], songdata.iloc[num][1]
    )

    return score

def add_score(songdata, num, current, destination, songs_left):
    current_a = songdata.loc[current][1]
    current_v = songdata.loc[current][0]
    destination_a = songdata.loc[destination][1]
    destination_v = songdata.loc[destination][0]
    step_a = (destination_a - current_a + .001) / songs_left
    step_v = (destination_v - current_v + .001) / songs_left

    dist_a_diff = songdata.iloc[num][1] - current_a - step_a
    dist_v_diff = songdata.iloc[num][0] - current_v - step_v

    score = dist_a_diff + dist_v_diff
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