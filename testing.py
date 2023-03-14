#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
import scipy as sp
import random
import pandas as pd
import seaborn as sns
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
from songdataset import SongDataset, SegmentDataset

QUADRANT_JSON   = "quadrants.json"
QUADRANT_CODES  = ["BL", "BR", "TL", "TR"]
QUADRANT_COMBOS = list(itertools.permutations(QUADRANT_CODES, 2))

DEEZER_STD_ALL  = "./data/deezer/deezer-std-all.csv"
DEEZER_PCA_ALL  = "./data/deezer/deezer-pca-all.csv"
DEEZER_PCA_SPO  = "./data/deezer/deezer-pca-spotify.csv"
DEEZER_PCA_MSD  = "./data/deezer/deezer-pca-msd.csv"
DEEZER_SEG_100  = "./data/deezer/deezer-segments-cnt100.csv"
DEEZER_SEG_D30  = "./data/deezer/deezer-segments-dur030.csv"

ARG_LENGTHS     = list(range(3, 20, 4))
ARG_DISTANCES   = [
     { "func": algos.cosine_score,      "name": "Cosine Similarity"}
    ,{ "func": algos.euclidean_score,   "name": "Euclidean Distance"}
    ,{ "func": algos.manhattan_score,   "name": "Manhattan Distance"}
    # ,{ "func": algos.minkowski3_score,  "name": "Minkowski Distance (order 3)"}
    ,{ "func": algos.jaccard_score,     "name": "Jaccard Distance"}
    ,{ "func": algos.mult_score,        "name": "Multiplied Ratios"}
    ,{ "func": algos.neighbors_rand,    "name": "Random Neighbors"}
]
ARG_NEIGHBORS_K = list(range(3, 32, 4))
ARG_SEGMENTS_CT = list(range(1, 200, 9))
ARG_SEGMENTS_DR = list(range(5, 60, 5))

DEF_LENGTHS     = 12
DEF_DISTANCES   = algos.cosine_score
DEF_NEIGHBORS_K = 7
DEF_SEGMENTS_CT = 100
DEF_SEGMENTS_DR = 30

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
            path=DEEZER_SEG_D30, knn=True, verbose=True,
            feat_index = 5, arousal = 4, valence = 3,
        )
    ]
    return arg_datasets

def load_samples(file = QUADRANT_JSON, count = 100):
    samples = {}
    while not os.path.exists(file) or not file.endswith(".json"):
        file = input("Sample JSON not found! Please enter a valid path: ")
    with open(file) as f:
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

    return point_combos

def pearson(playlistDF):
    points = playlistDF[["valence", "arousal"]].to_numpy()
    pr = np.corrcoef(points, rowvar=False)
    return abs(pr[1][0])

def spearman(playlistDF):
    points = playlistDF[["valence", "arousal"]].to_numpy()
    sr = sp.stats.spearmanr(points)
    return sr.correlation

def stepvar(playlistDF):
    steps = playlistDF["step"].to_numpy()[1:]
    sv = np.var(steps)
    return sv

def meansqr(playlistDF):
    points = playlistDF[["valence", "arousal"]].to_numpy()

    # For testing purposes: valence = x, arousal = y
    x, y = np.transpose(points)

    m = (y[-1] - y[0]) / (x[-1] - x[0])
    b = y[0] - m * x[0]

    y_pred = [-1 for i in range(len(x))]
    for i in range(len(points)):
        y_pred[i] = x[i] * m + b

    mse = ((y - y_pred)**2).mean()
    return mse

def feat_pearson(playlistDF, dataset):
    song_ids = playlistDF["id-deezer"].tolist()
    features = dataset.get_feats(song_ids)

    # Grab pairwise PCC of all features. NxN symmetric matrix.
    R = abs(features.corr('pearson').to_numpy())
    N = R.shape[0]

    # Get average of the sum of the triangle.
    trisum = (R.sum() - np.trace(R))
    triavg = trisum / (N * (N-1))
    return triavg

def feat_stepvar(playlistDF, dataset):
    song_ids = playlistDF["id-deezer"].tolist()
    features = dataset.get_feats(song_ids).to_numpy()

    # Calculate step sizes.
    steplist = np.empty(0)
    for i in range(1, features.shape[0]):
        a, b = features[i], features[i-1]
        norm = np.linalg.norm(a - b)
        steplist = np.append(steplist, norm / features.shape[1])
    
    sv = np.var(steplist)
    return sv

POINT_METRICS = [
    {"name": "Pearson correlation", "func": pearson},
    # {"name": "Spearman correlation", "func": spearman},
    {"name": "Step size variance", "func": stepvar},
    {"name": "Mean Square Error", "func": meansqr},
]

FEAT_METRICS = [
    {"name": "Pearson correlation", "func": feat_pearson},
    # {"name": "Spearman correlation", "func": spearman},
    {"name": "Step size variance", "func": feat_stepvar},
    # {"name": "Mean Square Error", "func": meansqr},
]

def evaluate(playlistDF, dataset, verbose=0):
    # evals = { "points": {}, "feats": {} }
    evals = {}
    
    if verbose >= 1: print("\nEvaluating points ...")
    for m in POINT_METRICS:
        val = m["func"](playlistDF)
        if verbose >= 1: print(m["name"], "\t", val)
        # evals["points"][m["name"]] = val
        evals[m["func"].__name__] = val

    if verbose >= 1: print("\nEvaluating features ...")
    for m in FEAT_METRICS:
        val = m["func"](playlistDF, dataset)
        if verbose >= 1: print(m["name"], "\t", val)
        # evals["feats"][m["name"]] = val    
        evals[m["func"].__name__] = val

    return evals

def metric_sheets(df, variable, dirname):
    for pm in POINT_METRICS:
        m = pm["func"].__name__
        desc = df.groupby(variable)[m].describe().round(6)
        desc.to_csv("{}/{}-{}.csv".format(dirname, variable, m))
    for fm in FEAT_METRICS:
        m = fm["func"].__name__
        desc = df.groupby(variable)[m].describe().round(6)
        desc.to_csv("{}/{}-{}.csv".format(dirname, variable, m))

def plot_scores(df, variable, dirname):
    for pm in POINT_METRICS:
        m = pm["func"].__name__
        plot.boxplots(df, m, variable, file="{}/{}-{}.png".format(dirname, variable, m))
    for fm in FEAT_METRICS:
        m = fm["func"].__name__
        plot.boxplots(df, m, variable, file="{}/{}-{}.png".format(dirname, variable, m))
