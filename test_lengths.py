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
import multiprocessing
import itertools

#our modules
import helper
import prodplay
import algos
import plot
import testing
from songdataset import SongDataset, SegmentDataset

def perQuadrant(oq, dq):
    qc = "{}{}".format(oq, dq)
    pairs = point_combos[qc]
    helper.makeDir("{}/{}".format(dirname, qc))

    for orig, dest in pairs:
        curdirname = "{}/{}/{}-{}".format(dirname, qc, orig, dest)
        helper.makeDir(curdirname)

        for variable in variables:
            # TODO: update default / variable arguments for each test.
            playlistDF = prodplay.makePlaylist(
                dataset, orig, dest, variable,
                score = testing.DEF_DISTANCES,
                neighbors = testing.DEF_NEIGHBORS_K,
                verbose = 0
            )

            # Save playlist DataFrame to CSV. TODO: update name.
            playlistDF.to_csv("{}/{}.csv".format(curdirname, str(variable)))

if __name__ == "__main__":
    samplecount = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    info = helper.loadConfig("config.json")

    # set up output directory. TODO: update for each test.
    dirname = helper.makeTestDir("lengths")

    # Points for testing.
    point_combos = testing.load_samples(testing.QUADRANT_JSON, samplecount)

    # Load datasets and variables. TODO: update for each test.
    dataset = SongDataset(
        name="Deezer+Spotify+MSD",
        cols=info["cols"]["deezer"] + info["cols"]["spotify"] + info["cols"]["msd"],
        path=testing.DEEZER_SPO_MSD, verbose=True, knn=True, 
    )
    variables = testing.ARG_LENGTHS

    # Run a process for each quadrant combo (12 in total).
    pQuadrants = multiprocessing.Pool(len(testing.QUADRANT_COMBOS))
    pQuadrants.starmap(perQuadrant, testing.QUADRANT_COMBOS)