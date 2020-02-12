#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# load song_id, average arousal and valence for each song from CSV to pandas DataFrame
# songdata = pd.read_csv("deam-data\\annotations\\annotations averaged per song\song_level\static_annotations_averaged_songs_1_2000.csv", header=0, index_col=0, usecols=[0, 1, 3])
songdata = pd.read_csv("deezer-data\\train.csv", header=0, index_col=0, usecols=[0, 3, 4])
print("read song data")
song_ids = list(songdata.index.values)

# train a KNN model 
neigh = NearestNeighbors()
neigh.fit(songdata.values)
print("trained data ... fingers crossed")

# input the starting and destination coordinates, set "current" to starting
# user_curr = int(input("Choose a number for a starting song: "))
# user_dest = int(input("Choose a number for an ending song: "))
user_curr = song_ids[random.randint(0, len(song_ids)) - 1]
user_dest = song_ids[random.randint(0, len(song_ids)) - 1]
print(songdata.loc[user_curr])
print(songdata.loc[user_dest])

# input the time needed to get from starting to destination (test cases here)
# teststart = int(input("How many minutes between your start and destination? "))
testcount = int(input("How many tests do you want to do? "))
teststart = 2
# testcount = 100

# array of smoothness values
smoothies = []

# array to store coordinates
coords = []

# for loop for testing different amounts of points in between
for time_reqd in range(teststart, teststart + testcount):
    current = user_curr
    destination = user_dest
    origin_a = songdata.loc[current][1]
    origin_v = songdata.loc[current][0]
    destination_a = songdata.loc[destination][1]
    destination_v = songdata.loc[destination][0]

    n_songs_reqd = time_reqd * 2 # multiply by 60 (seconds) and divide by 30 (seconds)

    sm_a_step = (destination_a - origin_a) / n_songs_reqd
    sm_v_step = (destination_v - origin_v) / n_songs_reqd

    # create a list of numbers (for songs) to store our "path" - let's call it song_list
    songlist = np.empty(0)
    songlist = np.append(songlist, current)

    # testing variable for running a single iteration of the while loop
    test = True 
    smooth_steps = []
    


    # while the current song isn't the destination song and the number of songs required isn't met
    while ((current != destination) & (len(songlist) < n_songs_reqd)):
    # while (test):
        # grab the a_dist and v_dist between current and destination (replace with list for scaling dimensions)
        current_a = songdata.loc[current][1]
        current_v = songdata.loc[current][0]

        # was gonna do slope and distance, but what if the imagined vector was in the OPPOSITE direction?
        distance = np.sqrt(np.square(destination_a - current_a) + np.square(destination_v - current_v))
        dist_a = destination_a - current_a + .001
        dist_v = destination_v - current_v + .001

        # divide distance by the n_songs_reqd to get a step size
        step_size = distance / (n_songs_reqd - len(songlist))
        step_a = dist_a / (n_songs_reqd - len(songlist))
        step_v = dist_v / (n_songs_reqd - len(songlist))

        # grab the nearest neighbors within 110% * step_size of the current point
        r_neighbors = neigh.radius_neighbors([songdata.loc[current].array], radius=(1.1 * step_size))
        candidates = np.array(r_neighbors[1])
        candidates = candidates[0]

        for i in range(len(songlist)):
            candidates = candidates[candidates != songdata.index.get_loc(songdata.loc[songlist[i]].name)]
        
        if (len(candidates) < 1):
            k_neighbors = neigh.kneighbors([songdata.loc[current].array], n_songs_reqd + 1)
            candidates = np.array(k_neighbors[1])
            candidates = candidates[0]

            for i in range(len(songlist)):
                candidates = candidates[candidates != songdata.index.get_loc(songdata.loc[songlist[i]].name)]

        cand_scores = []
        cand_smooths = []

        for i in range(len(candidates)):
            num = candidates[i]
            
            ## RATIO METHOD
            # dist_a_ratio = (songdata.iloc[num][1] - current_a) / step_a
            # dist_v_ratio = (songdata.iloc[num][0] - current_v) / step_v
            # cand_scores.append(np.absolute(1 - dist_a_ratio) * np.absolute(1 - dist_v_ratio))

            ## DIFFERENCE METHOD
            dist_a_diff = songdata.iloc[num][1] - current_a - step_a
            dist_v_diff = songdata.iloc[num][0] - current_v - step_v

            smooth_a_diff = songdata.iloc[num][1] - (origin_a + ((len(songlist) - 1) * sm_a_step))
            smooth_v_diff = songdata.iloc[num][0] - (origin_v + ((len(songlist) - 1) * sm_v_step))
            
            added_dist = dist_a_diff + dist_v_diff
            euclid_dist = np.sqrt(np.square(dist_a_diff) + np.square(dist_v_diff))
            euclid_smooth = np.sqrt(np.square(smooth_a_diff) + np.square(smooth_v_diff))

            cand_scores.append(euclid_dist)
            cand_smooths.append(euclid_smooth)

        # select the song which has the ratio/difference closest to 1.00 to be the new value of "current"
        min_indices = np.argmin(cand_scores)
        min_cand_smooth = cand_smooths[min_indices]
        smooth_steps.append(min_cand_smooth)

        min_cand_index = candidates[min_indices]
        current = song_ids[min_cand_index]
        songlist = pd.unique(np.append(songlist, current))
        test = False

    smoothie = np.mean(smooth_steps)
    smoothies.append(smoothie)
    print("{}: {}".format(time_reqd, smoothie))

    v_points = []
    a_points = []

    for i in range(len(songlist)):
        v_points.append(songdata.loc[songlist[i]][1])
        a_points.append(songdata.loc[songlist[i]][0])

    wrapped_points = [v_points, a_points]
    coords.append(wrapped_points)

plt.xlabel('valence')
plt.ylabel('arousal')
plt.legend(smoothies)

smoothest = np.argmin(smoothies)
print(smoothest + teststart)

for i in range(len(coords)):
    plt.plot(coords[i][0], coords[i][1])
plt.show()

plt.plot(coords[smoothest][0], coords[smoothest][1])
plt.show()

plt.xlabel('# of tests')
plt.ylabel('smoothness of paths')
plt.plot(smoothies)
plt.show()