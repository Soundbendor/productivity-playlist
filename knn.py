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
origin = int(input("Choose a number for a starting song: "))
destination = int(input("Choose a number for an ending song: "))
current = origin
print(songdata.loc[current, :]) #TODO: check if the number is in the list of indexes or else errors will happen (if you want)
print(songdata.loc[destination, :])

# input the time needed to get from starting to destination
# since these are 30 second song clips, divide the time to figure out how many songs we need (n_songs_needed) (round down if necessary)
time_reqd = int(input("How many minutes between your start and destination? "))
n_songs_reqd = 

# create a list of numbers (for songs) to store our "path" - let's call it song_list
# 
# while (current != destination && n_songs_needed > 0):
#   grab the slope and distance between current and destination ... can be replaced with a vector when we deal with more than 2 dimensions
#   divide distance by the n_songs_needed to get a step size
#   n_songs_needed -= 1
#   grab the K nearest neighbors of the current point
#   calculate the slopes and distances between them
#   calculate the between between each of those and the desired slope and distance
#   put "current" in "song_list"
#   select the song which has the ratio closest to 1.00 to be the new value of "current"