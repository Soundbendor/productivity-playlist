#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
import scipy as sp
import random
import pandas as pd
import matplotlib.pyplot as plt
import json
import pprint
import time
import sys
import os
import math
import warnings
import itertools

#our modules
import helper
import prodplay
import spotify
import plot
import algos
from songdataset import SongDataset
from segmentdataset import SegmentDataset

QUADRANT_JSON   = "./quadrants/std-22-05-03_1229/songs.json"
QUADRANT_CODES  = ["BL", "BR", "TL", "TR"]
QUADRANT_COMBOS = list(itertools.permutations(QUADRANT_CODES, 2))

DEEZER_STD_ALL  = "./data/deezer/deezer-std-all.csv"
DEEZER_PCA_ALL  = "./data/deezer/deezer-pca-all.csv"
DEEZER_PCA_SPO  = "./data/deezer/deezer-pca-spotify.csv"
DEEZER_PCA_MSD  = "./data/deezer/deezer-pca-msd.csv"
DEEZER_SEG_100  = "./data/deezer/deezer-segments-cnt100.csv"
DEEZER_SEG_D30  = "./data/deezer/deezer-segments-dur030.csv"

ARG_NEIGHBORS_K = list(range(5, 30, 2))
ARG_NEIGHBORS_R = list(range(0.05, 0.55, 0.05))
ARG_SEGMENTS_CT = list(range(1, 100, 3))
ARG_LENGTHS     = list(range(3, 20))
ARG_DISTANCES   = [
     { "func": algos.cosine_score,      "name": "Cosine Similarity"}
    ,{ "func": algos.euclidean_score,   "name": "Euclidean Distance"}
    ,{ "func": algos.manhattan_score,   "name": "Manhattan Distance"}
    ,{ "func": algos.minkowski3_score,  "name": "Minkowski Distance (order 3)"}
    ,{ "func": algos.jaccard_score,     "name": "Jaccard Distance"}
    ,{ "func": algos.mult_score,        "name": "Multiplied Ratios"}
    ,{ "func": algos.neighbors_rand,    "name": "Random Neighbors"}
]

def LOAD_DATASETS(cols):
    arg_datasets = [
        SongDataset(
            name="Deezer",
            cols=cols["deezer"],
            path=DEEZER_STD_ALL, knn=True, verbose=True,
            feat_index = 3, arousal = 4, valence = 3,
        ),
        SongDataset(
            name="Deezer+Spotify",
            cols=cols["deezer"] + cols["spotify"],
            path=DEEZER_STD_ALL, knn=True, verbose=True,
            feat_index = 5, arousal = 4, valence = 3,
        ),
        SongDataset(
            name="Deezer+MSD",
            cols=cols["deezer"] + cols["msd"],
            path=DEEZER_STD_ALL, knn=True, verbose=True,
            feat_index = 5, arousal = 4, valence = 3,
        ),
        SongDataset(
            name="Deezer+Spotify+MSD",
            cols=cols["deezer"] + cols["spotify"] + cols["msd"],
            path=DEEZER_STD_ALL, knn=True, verbose=True,
            feat_index = 5, arousal = 4, valence = 3,
        ),
        SongDataset(
            name="PCA-Deezer+Spotify",
            path=DEEZER_PCA_SPO, 
            knn=True, verbose=True,
            feat_index = 5, arousal = 4, valence = 3,
        ),
        SongDataset(
            name="PCA-Deezer+MSD",
            path=DEEZER_PCA_MSD, 
            knn=True, verbose=True,
            feat_index = 5, arousal = 4, valence = 3,
        ),
        SongDataset(
            name="PCA-Deezer+Spotify+MSD",
            path=DEEZER_PCA_ALL, 
            knn=True, verbose=True,
            feat_index = 5, arousal = 4, valence = 3,
        ),
        SegmentDataset(
            name="Deezer+Segments-100cnt",
            cols=cols["deezer"] + cols["segments"],
            path=DEEZER_SEG_100, knn=True, verbose=True,
            feat_index = 5, arousal = 4, valence = 3,
        ),
        SegmentDataset(
            name="Deezer+Segments-030sec",
            cols=cols["deezer"] + cols["segments"],
            path=DEEZER_SEG_100, knn=True, verbose=True,
            feat_index = 5, arousal = 4, valence = 3,
        )
    ]
    return arg_datasets

def load_samples(file = QUADRANT_JSON, count = 100):
    samples = {}
    while not os.path.exists(samplejson) or not samplejson.endswith(".json"):
        samplejson = input("Sample JSON not found! Please enter a valid path: ")
    with open(samplejson) as f:
        samples = json.load(f)
        print("Sample file loaded!")    

    point_combos = {}
    print("\nLoading point combos!")
    for a, b in QUADRANT_COMBOS:
        pairname = "{}{}".format(a, b)
        if pairname in point_combos:
            continue
        point_combos[pairname] = []
        print("- {} ... ".format(pairname), end='')

        for i, j in itertools.product(range(count), repeat=2):
            point_combos[pairname].append((int(samples[a][i]), int(samples[b][j])))
        
        print("Loaded!")

    return samples

def pearson(playlistDF):
    points = playlistDF[["valence", "arousal"]].to_numpy()
    pr = np.corrcoef(points, rowvar=False)
    return pr[1][0]

def spearman(playlistDF):
    points = playlistDF[["valence", "arousal"]].to_numpy()
    sr = sp.stats.spearmanr(points)
    return sr.correlation

def stepvar(playlistDF):
    steps = playlistDF["evenness"].to_numpy()[1:]
    sv = np.var(steps)
    return sv

def meansquare(playlistDF):
    points = playlistDF[["valence", "arousal"]].to_numpy()

    orig, dest = points[0], points[-1]
    stepcount = len(points) - 1
    
    diff = dest - orig
    step = diff / stepcount
    target = np.array([(orig + (i * step)) for i in range(1, stepcount)])

    total_mse = 0
    for i in range(1, stepcount):
        mse_av = (points[i] - target[i-1])**2
        mse_unit = np.sqrt(sum(mse_av))
        total_mse = total_mse + mse_unit
    
    return total_mse / stepcount