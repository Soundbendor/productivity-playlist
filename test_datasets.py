# import necessary modules
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
from pprint import pprint
import time
import sys
import os
import math
import itertools

#our modules
import helper
import prodplay
import algos
import testing
from songdataset import SongDataset

# Some constants good to figure out now
datasheet   = "./data/deezer/deezer-std-all.csv"
samplejson  = "./quadrants/std-22-05-03_1229/songs.json"
samplecount = int(sys.argv[1]) if len(sys.argv) > 1 else 100
info = helper.loadConfig("config.json")

# set up output directories
dirname = helper.makeTestDir("datasets")

# Load random sample points.
samples = {}
while not os.path.exists(samplejson) or not samplejson.endswith(".json"):
    samplejson = input("Sample JSON file not found! Please enter a valid path: ")
with open(samplejson) as f:
    samples = json.load(f)
    print("Sample file loaded!")

# create random sample pairs
quadrants = ["BL", "BR", "TL", "TR"]
point_combos = {}

# Use itertools to generate quadrant combos:
    # product if we want inner quadrants too
    # permutations if we want both ways
    # comdinations if we only care about one way
quadrant_combos = list(itertools.permutations(quadrants, 2))

print("\nLoading point combos!")
for a, b in quadrant_combos:
    pairname = "{}{}".format(a, b)
    if pairname in point_combos:
        continue
    point_combos[pairname] = []
    print("- {} ... ".format(pairname), end='')

    for i, j in itertools.product(range(samplecount), repeat=2):
        point_combos[pairname].append((int(samples[a][i]), int(samples[b][j])))
    
    print("Loaded!")

# Let's create an array of the song datasets.
print("\nLoading datasets.")
datasets = [
    SongDataset(
        name="Deezer",
        cols=info["cols"]["deezer"],
        path=datasheet, knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="Deezer+Spotify",
        cols=info["cols"]["deezer"] + info["cols"]["spotify"],
        path=datasheet, knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="Deezer+MSD",
        cols=info["cols"]["deezer"] + info["cols"]["msd"],
        path=datasheet, knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="Deezer+Spotify+MSD",
        cols=info["cols"]["deezer"] + info["cols"]["spotify"] + info["cols"]["msd"],
        path=datasheet, knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="PCA-Deezer+Spotify",
        path="deezerpca-spotify.csv", 
        knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="PCA-Deezer+MSD",
        path="deezerpca-msd.csv", 
        knn=True, verbose=True, start_index=3
    ),
    SongDataset(
        name="PCA-Deezer+Spotify+MSD",
        path="deezerpca-spotify+msd.csv", 
        knn=True, verbose=True, start_index=3
    )
]

# TODO?: test at various lengths
length = 10

# TODO?: test at various neighbor counts
neighbors = 7

# TODO: Columns for our result sheets

# For each dataset and point combination:
for dataset in datasets:
    print("Testing {}".format(dataset.name))
    helper.makeDir("{}/{}".format(dirname, dataset.name))

    # collect table of results
    # results = {}
    # for col in resultcols: results[col] = []

    # For each point combination:
    for oq, dq in quadrant_combos:
        qc = "{}{}".format(oq, dq)
        pairs = point_combos[qc]
        print(" - {}".format(qc))

        curdirname = "{}/{}/{}".format(dirname, dataset.name, qc)
        helper.makeDir(curdirname)

        # use cosine similarity and euclidean distance
        # use mean squared error for smoothness
        # use variance of step sizes for even steps
        for orig, dest in pairs:

            # Generate playlists with each distance
            playlistDF = prodplay.makePlaylist(dataset, orig, dest, length)

            # Evaluate playlist with each metric

            # Add results to our collection

        
