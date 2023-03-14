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
from songdataset import SongDataset

def perVariable(dataset, orig, dest, l, score, k, outfile):
    playlistDF = prodplay.makePlaylist(
        dataset, orig, dest, l,
        score = score,
        neighbors = k,
        verbose = 0
    )
    playlistDF.to_csv(outfile)
    return

def perQuadCombo(orig, dest, datasets, dirname):
    curdirname = "{}/{}-{}".format(dirname, orig, dest)
    helper.makeDir(curdirname)
    # qc = dirname.split("-")[-1]
    # print(f"{qc} ... {orig} -> {dest}")

    # argVariable = [(
    #     d, orig, dest, testing.DEF_LENGTHS, 
    #     testing.DEF_DISTANCES, testing.DEF_NEIGHBORS_K,
    #     "{}/{}.csv".format(curdirname, d.name)
    # ) for d in datasets]

    # p = multiprocessing.Pool(len(datasets))
    # p.starmap(perVariable, argVariable)

    # for orig, dest in pairs:
    for dataset in datasets:
        # Name of metric in string form. TODO: update for each test type.
        name = dataset.name
        
        # Generate playlist with this dataset and default other arguments.
        # TODO: update default / variable arguments for each test.
        playlistDF = prodplay.makePlaylist(
            dataset, orig, dest, testing.DEF_LENGTHS,
            score = testing.DEF_DISTANCES,
            neighbors = testing.DEF_NEIGHBORS_K,
            verbose = 0
        )

        # Save playlist DataFrame to LaTeX.
        playlistDF.to_csv("{}/{}.csv".format(curdirname, name))

    return

def perQuadrant(oq, dq):
    qc = "{}{}".format(oq, dq)
    pairs = point_combos[qc]
    # print(qc)
    helper.makeDir("{}/{}".format(dirname, qc))

    # argQuadCombo = [(
    #     orig, dest, datasets, "{}/{}".format(dirname, qc)
    # ) for (orig, dest) in pairs]

    # pQuadCombo = multiprocessing.Pool(len(pairs))
    # pQuadCombo.starmap(perQuadCombo, argQuadCombo)    

    for idx, (orig, dest) in enumerate(pairs):
        print(f"{qc} ... {idx + 1} / {len(pairs)}\t")
        perQuadCombo(orig, dest, datasets, "{}/{}".format(dirname, qc))

if __name__ == "__main__":
    # Some constants good to figure out now
    # samplejson  = "./ismir2022/quadrants/std-22-05-03_1229/songs.json"
    samplecount = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(os.getcwd())
    info = helper.loadConfig(f"{os.getcwd()}/config.json")

    # set up output directories
    variable = "dataset"
    dirname = helper.makeTestDir(f"{variable}s")

    # Points for testing.
    point_combos = testing.load_samples(testing.QUADRANT_JSON, samplecount)

    # Let's create an array of the song datasets.
    # TODO: for other tests, only load default dataset.
    print("\nLoading datasets.")
    datasets = testing.LOAD_DATASETS(info["cols"])

    print(testing.QUADRANT_COMBOS)
    pQuadrants = multiprocessing.Pool(len(testing.QUADRANT_COMBOS))
    pQuadrants.starmap(perQuadrant, testing.QUADRANT_COMBOS)

    # # For each quadrant:
    # for oq, dq in testing.QUADRANT_COMBOS:
    #     qc = "{}{}".format(oq, dq)
    #     pairs = point_combos[qc]
    #     print(qc)
    #     helper.makeDir("{}/{}".format(dirname, qc))

    #     argQuadCombo = [(
    #         orig, dest, datasets, "{}/{}".format(dirname, qc)
    #     ) for (orig, dest) in pairs]

    #     pQuadCombo = multiprocessing.Pool(len(pairs))
    #     pQuadCombo.starmap(perQuadCombo, argQuadCombo)

        # # For each point combination:
        # for idx, (orig, dest) in enumerate(pairs):
        #     curdirname = "{}/{}/{}-{}".format(dirname, qc, orig, dest)
        #     helper.makeDir(curdirname)

        #     argVariable = [(
        #         d, orig, dest, testing.DEF_LENGTHS, 
        #         testing.DEF_DISTANCES, testing.DEF_NEIGHBORS_K,
        #         "{}/{}.csv".format(curdirname, d.name)
        #     ) for d in datasets]

        #     pVariable = multiprocessing.Pool(len(datasets))
        #     pVariable.starmap(perVariable, argVariable)
        #     print(f"{qc} ... {idx + 1} / {len(pairs)}\t", end="\r")

            # for orig, dest in pairs:
            # for dataset in datasets:
            #     # Name of metric in string form. TODO: update for each test type.
            #     name = dataset.name
                
            #     # Generate playlist with this dataset and default other arguments.
            #     # TODO: update default / variable arguments for each test.
            #     playlistDF = prodplay.makePlaylist(
            #         dataset, orig, dest, testing.DEF_LENGTHS,
            #         score = testing.DEF_DISTANCES,
            #         neighbors = testing.DEF_NEIGHBORS_K,
            #         verbose = 0
            #     )

            #     # Save playlist DataFrame to LaTeX.
            #     playlistDF.to_csv("{}/{}.csv".format(curdirname, name))
