import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def euclideanDistance(x1, y1, x2, y2):
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

def get_candidates(songdata, current, destination, n_songs_reqd, songlist, neigh):
    destination_a = songdata.loc[destination][1]
    destination_v = songdata.loc[destination][0]
    current_a = songdata.loc[current][1]
    current_v = songdata.loc[current][0]

    dist_a = destination_a - current_a + .001
    dist_v = destination_v - current_v + .001

    songs_left = n_songs_reqd - len(songlist)
    radius = 1.1 * np.sqrt(np.square(dist_a) + np.square(dist_v)) / songs_left
    
    r_neighbors = neigh.radius_neighbors([songdata.select_dtypes(include='float64').loc[current].array], radius=radius)
    candidates = np.array(r_neighbors[1])
    candidates = candidates[0]

    for i in range(len(songlist)):
        candidates = candidates[candidates != songdata.index.get_loc(songdata.loc[songlist[i]].name)]
    
    if (len(candidates) < 1):
        k_neighbors = neigh.kneighbors([songdata.select_dtypes(include='float64').loc[current].array], n_songs_reqd + 1)
        candidates = np.array(k_neighbors[1])
        candidates = candidates[0]

        for i in range(len(songlist)):
            candidates = candidates[candidates != songdata.index.get_loc(songdata.loc[songlist[i]].name)]
    
    return candidates

def choose_candidate(songdata, candidates, current, origin, destination, n_songs_reqd, songs_so_far):
    current_a = songdata.loc[current][1]
    current_v = songdata.loc[current][0]
    origin_a = songdata.loc[origin][1]
    origin_v = songdata.loc[origin][0]
    destination_a = songdata.loc[destination][1]
    destination_v = songdata.loc[destination][0]
    
    smooth_a_step = (destination_a - origin_a) / n_songs_reqd
    smooth_v_step = (destination_v - origin_v) / n_songs_reqd
    
    step_a = (destination_a - current_a + .001) / (n_songs_reqd - songs_so_far)
    step_v = (destination_v - current_v + .001) / (n_songs_reqd - songs_so_far)
    
    cand_scores = []
    cand_smooths = []
    
    # TEST ALL THE CANDIDATES FOR THE NEXT SONG TO FIND THE BEST ONE
    for i in range(len(candidates)):
        num = candidates[i]

        cand_scores.append(euclideanDistance(
            current_v + step_v,
            current_a + step_a, 
            songdata.iloc[num][0],
            songdata.iloc[num][1]
        ))

        cand_smooths.append(euclideanDistance(
            origin_v + (songs_so_far * smooth_v_step),
            origin_a + (songs_so_far * smooth_a_step),
            songdata.iloc[num][0],
            songdata.iloc[num][1]
        ))

    # select the song which has the score closest to 0 to be the new value of "current"
    min_score_index = np.argmin(cand_scores)
    min_cand_smooth = cand_smooths[min_score_index]
    min_cand_song = candidates[min_score_index]
    
    return min_cand_song, min_cand_smooth

def makePlaylist(songdata, song_ids, origin, destination, n_songs_reqd, neigh):
    # create a list of numbers (for songs) to store our "path" - let's call it song_list
    songlist = np.empty(0)
    songlist = np.append(songlist, origin)
    
    #storing the "smoothness" values for each song in the playlist
    smooth_steps = []

    current = origin
    # while the current song isn't the destination song and the number of songs required isn't met
    while ((current != destination) & (len(songlist) < n_songs_reqd)):        
        # get the neighbors of the current song based on the size of the next step
        candidates = get_candidates(songdata, current, destination, n_songs_reqd, songlist, neigh)
        
        # choose the song that has the closest arousal and valence distances to the desired values
        next_song, next_smooth = choose_candidate(
            songdata, candidates, current, origin, destination, n_songs_reqd, len(songlist)
        )
        
        # grab the song index and smoothness factor (for testing use) and put into appropriate lists
        smooth_steps.append(next_smooth)
        current = song_ids[next_song]
        songlist = pd.unique(np.append(songlist, current))

    # return the song list and the average smoothness of the songs in the playlist
    smoothie = np.mean(smooth_steps)
    return songlist, smoothie