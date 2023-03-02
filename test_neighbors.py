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
from segmentdataset import SegmentDataset

# Some constants good to figure out now
samplejson  = "./ismir2022/quadrants/std-22-05-03_1229/songs.json"
samplecount = int(sys.argv[1]) if len(sys.argv) > 1 else 100
info = helper.loadConfig("config.json")

# set up output directories
dirname = helper.makeTestDir("neighbors")

# Points for testing.
point_combos = testing.load_samples(samplejson, samplecount)

# Let's create an array of the song datasets.
# TODO: for other tests, only load default dataset.
print("\nLoading datasets.")
dataset = SegmentDataset(
    name="Deezer+Segments-100cnt",
    cols=info["cols"]["deezer"] + info["cols"]["segments"],
    path=testing.DEEZER_SEG_100, knn=True, verbose=True,
    feat_index = 5, arousal = 4, valence = 3,
)

# Columns for our result sheets
resultcols = ["oq", "dq", "orig", "dest"]
for pm in testing.POINT_METRICS:
    resultcols.append(pm["func"].__name__)
for fm in testing.FEAT_METRICS:
    resultcols.append(fm["func"].__name__)

# For each K and point combination:
# TODO: change what gets iterated thru for each test.
for k in testing.ARG_NEIGHBORS_K:
    print("\nTesting{}".format(k))
    helper.makeDir("{}/{}".format(dirname, k))

    # collect table of results
    results = {}
    for col in resultcols: results[col] = []

    # For each point combination:
    for oq, dq in testing.QUADRANT_COMBOS:
        qc = "{}{}".format(oq, dq)
        pairs = point_combos[qc]
        print(" - {}".format(qc))

        curdirname = "{}/{}/{}".format(dirname, k, qc)
        helper.makeDir(curdirname)

        for orig, dest in pairs:

            # Generate playlist with this K and default other arguments.
            # TODO: update default / variable arguments for each test.
            playlistDF = prodplay.makePlaylist(
                dataset, orig, dest, testing.DEF_LENGTHS,
                score = testing.DEF_DISTANCES,
                neighbors = k,
                verbose = 0
            )

            # Save playlist DataFrame to LaTeX.
            playlistDF.to_csv("{}/{}-{}.csv".format(curdirname, orig, dest))

            # Add results to our collection
            results["oq"].append(oq)
            results["dq"].append(dq)
            results["orig"].append(orig)
            results["dest"].append(dest)

            # Evaluate playlist with each metric
            evals = testing.evaluate(playlistDF, dataset)
            for key in evals:
                results[key].append(evals[key])

    resultDF = pd.DataFrame(results)
    resultDF.to_csv("{}/{}/results.csv".format(dirname, k))