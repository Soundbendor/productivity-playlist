#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# load the annotation data from CSVs to ... data frames?
# (specifically the song_id, average arousal and valence for each song)
# (this line just loads the first 2000 songs ... I figured that'd be enough for a first run)
songdata = pd.read_csv("deam-data\\annotations\\annotations averaged per song\song_level\static_annotations_averaged_songs_1_2000.csv", header=0, index_col=0, usecols=[0, 1, 3])
print("read song data")
print(songdata.loc[156, :])

# graphing around here wouldn't be a terrible idea

# train a KNN model ??? 
neigh = NearestNeighbors()
neigh.fit(songdata.values)
print("trained data ... fingers crossed")

# input the starting and destination coordinates, set "current" to starting
current = int(input("Choose a number for a starting song: "))
destination = int(input("Choose a number for an ending song: "))

print(songdata.loc[current, :]) #TODO: check if the number is in the list of indexes or else errors will happen (if you want)
print(songdata.loc[destination, :])

destination_a = songdata.loc[destination][0]
destination_v = songdata.loc[destination][1]

# input the time needed to get from starting to destination
# since these are 30 second song clips, divide the time to figure out how many songs we need (n_songs_reqd) (round down if necessary)
time_reqd = int(input("How many minutes between your start and destination? "))
n_songs_reqd = time_reqd * 2 # multiply by 60 (seconds) and divide by 30 (seconds)

# create a list of numbers (for songs) to store our "path" - let's call it song_list
songlist = []

# while the current song isn't the destination song, and there are no songs left
while ((current != destination) & (n_songs_reqd > 0)):
    # put "current" in "song_list"
    songlist.append(current)

    # grab the a_dist and v_dist between current and destination ... 
    # (can be replaced with a vector when we deal with more than 2 dimensions)
    current_a = songdata.loc[current][0]
    current_v = songdata.loc[current][1]

    # was gonna do slope and distance, but what if the imagined vector was in the OPPOSITE direction?
    distance = np.sqrt(np.square(destination_a - current_a) + np.square(destination_v - current_v))
    slope = (destination_a - current_a) / (destination_v - current_v)

    dist_a = destination_a - current_a
    dist_v = destination_v - current_v

    # divide distance by the n_songs_reqd to get a step size
    step_size = distance / n_songs_reqd
    step_a = dist_a / n_songs_reqd
    step_v = dist_v / n_songs_reqd
    n_songs_reqd = n_songs_reqd - 1

    # grab the nearest neighbors within 110% * step_size of the current point
    candidates = neigh.radius_neighbors([songdata.loc[current].array], radius=(1.1 * step_size))

    # calculate the relative a_dists and v_dists between them and current
    # calculate the ratio between each of those and the desired a_dist and v_dist
    print(candidates[1])
    dists_a = []
    dists_v = []
    combined = []

    for i in range(len(candidates[1])):
        num = candidates[1][i]
        entry = songdata.loc[num, :]
        # print(songdata.loc[num])
        print(num)

        dist_a_ratio = (songdata.loc[num].to_numpy()[0] - current_a) / dist_a
        dist_v_ratio = (songdata.loc[num].to_numpy()[1] - current_v) / dist_v

        dists_a.append(np.absolute(1 - dist_a_ratio))
        dists_v.append(np.absolute(1 - dist_v_ratio))
        combined.append(dists_a[i] * dists_v[i])

    # select the song which has the ratio closest to 1.00 to be the new value of "current"
    min_indices = np.argmin(combined)
    print(min_indices)
    min_song_id = candidates[1][0][min_indices]
    print(songdata.loc[min_song_id])
