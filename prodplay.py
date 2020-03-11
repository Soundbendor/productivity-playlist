import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import algos

def get_candidates(songdata, current, destination, n_songs_reqd, songlist, model):
    destination_a = songdata.loc[destination][1]
    destination_v = songdata.loc[destination][0]
    current_a = songdata.loc[current][1]
    current_v = songdata.loc[current][0]

    dist_a = destination_a - current_a + .001
    dist_v = destination_v - current_v + .001

    songs_left = n_songs_reqd - len(songlist) + 1
    radius = 1.1 * np.sqrt(np.square(dist_a) + np.square(dist_v)) / songs_left
    
    target_a = current_a + (dist_a / songs_left)
    target_v = current_v + (dist_v / songs_left)
    target = [[target_v, target_a]]
    
    # r_neighbors = model.radius_neighbors(target, radius=radius)
    r_neighbors = model.kneighbors(target, n_neighbors=19)
    candidates = np.array(r_neighbors[1])
    candidates = candidates[0]

    for i in range(len(songlist)):
        candidates = candidates[candidates != songdata.index.get_loc(songdata.loc[songlist[i]].name)]
    
    if (len(candidates) < 1):
        t_neighbors = model.kneighbors(target, n_neighbors=9)
        candidates = np.array(t_neighbors[1])
        candidates = candidates[0]
        
        if (len(candidates) < 1):
            k_neighbors = model.kneighbors(target, n_neighbors = n_songs_reqd + 1)
            print(k_neighbors)
            candidates = np.array(k_neighbors[1])
            candidates = candidates[0]

            for i in range(len(songlist)):
                candidates = candidates[candidates != songdata.index.get_loc(songdata.loc[songlist[i]].name)]

    return candidates

def choose_candidate(songdata, candidates, current, origin, destination, n_songs_reqd, songs_so_far, score):
    smooth_a_step = (songdata.loc[destination][1] - songdata.loc[origin][1]) / n_songs_reqd
    smooth_v_step = (songdata.loc[destination][0] - songdata.loc[origin][0]) / n_songs_reqd
    songs_left = n_songs_reqd - songs_so_far 
    
    cand_scores = []
    cand_smooths = []
    
    # TEST ALL THE CANDIDATES FOR THE NEXT SONG TO FIND THE BEST ONE
    for i in range(len(candidates)):
        num = candidates[i]

        cand_scores.append(score(songdata, num, current, destination, songs_left))

        cand_smooths.append(algos.euclid_dist(
            songdata.loc[origin][0] + (songs_so_far * smooth_v_step),
            songdata.loc[origin][1] + (songs_so_far * smooth_a_step),
            songdata.iloc[num][0],
            songdata.iloc[num][1]
        ))

    # select the song which has the score closest to 0 to be the new value of "current"
    min_score_index = np.argmin(cand_scores)
    min_cand_smooth = cand_smooths[min_score_index]
    min_cand_song = candidates[min_score_index]
    
    return min_cand_song, min_cand_smooth

def makePlaylist(songdata, origin, destination, n_songs_reqd, model, score):
    song_ids = list(songdata.index.values)
   
    # create a list of numbers (for songs) to store our "path" - let's call it song_list
    songlist = np.empty(0)
    songlist = np.append(songlist, origin)
    
    #storing the "smoothness" values for each song in the playlist
    smooth_steps = []

    current = origin
    # while the current song isn't the destination song and the number of songs required isn't met
    while ((current != destination) & (len(songlist) - 1 < n_songs_reqd)):        
        # get the neighbors of the current song based on the size of the next step
        candidates = get_candidates(songdata, current, destination, n_songs_reqd, songlist, model)

        if (score != algos.rand_score):
            # choose the song that has the closest arousal and valence distances to the desired values
            next_song, next_smooth = choose_candidate(
                songdata, candidates, current, origin, destination, n_songs_reqd, len(songlist) - 1, score
            )
        
        else:
            next_song, next_smooth = score(
                songdata, candidates, destination, origin, n_songs_reqd, len(songlist)
            )

        # grab the song index and smoothness factor (for testing use) and put into appropriate lists
        smooth_steps.append(next_smooth)
        current = song_ids[next_song]
        songlist = pd.unique(np.append(songlist, current))


    # for some reason, it will fill up the length of the list w/o getting to the final song
    songlist = pd.unique(np.append(songlist, destination))

    # return the song list and the average smoothness of the songs in the playlist
    smoothie = np.mean(smooth_steps)
    return songlist, smoothie