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
import plot
import testing
from songdataset import SongDataset

# Some constants good to figure out now
samplejson  = "./ismir2022/quadrants/std-22-05-03_1229/songs.json"
samplecount = int(sys.argv[1]) if len(sys.argv) > 1 else 100
info = helper.loadConfig("config.json")

# set up output directories
dirname = helper.makeTestDir("datasets")

# Points for testing.
point_combos = testing.load_samples(samplejson, samplecount)

# Let's create an array of the song datasets.
print("\nLoading datasets.")
datasets = testing.LOAD_DATASETS(info["cols"])

# TODO: Columns for our result sheets
resultcols = ["oq", "dq", "orig", "dest", "pearson", "stepvar", "meansqr"]

# For each dataset and point combination:
for dataset in datasets:
    print("Testing {}".format(dataset.name))
    helper.makeDir("{}/{}".format(dirname, dataset.name))

    # collect table of results
    results = {}
    for col in resultcols: results[col] = []

    # For each point combination:
    for oq, dq in testing.QUADRANT_COMBOS:
        qc = "{}{}".format(oq, dq)
        pairs = point_combos[qc]
        print(" - {}".format(qc))

        curdirname = "{}/{}/{}".format(dirname, dataset.name, qc)
        helper.makeDir(curdirname)

        for orig, dest in pairs:

            # Generate playlists with each distance
            playlistDF = prodplay.makePlaylist(
                dataset, orig, dest, testing.DEF_LENGTHS,
                score = testing.DEF_DISTANCES,
                neighbors = testing.DEF_NEIGHBORS_K,
                radius = testing.DEF_NEIGHBORS_R,
                mode = "k",
                verbose = 0
            )

            # Evaluate playlist with each metric
            pearson = testing.pearson(playlistDF)
            stepvar = testing.stepvar(playlistDF)
            meansqr = testing.meansqr(playlistDF)

            # Save playlist DataFrame to LaTeX.
            playlistDF.to_csv("{}/{}-{}.csv".format(curdirname, orig, dest))

            # Add results to our collection
            results["oq"].append(oq)
            results["dq"].append(dq)
            results["orig"].append(orig)
            results["dest"].append(dest)
            results["pearson"].append(pearson)
            results["stepvar"].append(stepvar)
            results["meansqr"].append(meansqr)

    resultDF = pd.DataFrame(results)
    resultDF.to_csv("{}/{}/results.csv".format(dirname, dataset.name))