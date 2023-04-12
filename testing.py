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

DEEZER_DIR = "./data/deezer/powert" 

DEEZER_SPO_MSD  = f"{DEEZER_DIR}/Deezer+Spotify+MSD.csv"
DEEZER_PCA_ALL  = f"{DEEZER_DIR}/PCA-Deezer+Spotify+MSD.csv"
DEEZER_PCA_SPO  = f"{DEEZER_DIR}/PCA-Deezer+Spotify.csv"
DEEZER_PCA_MSD  = f"{DEEZER_DIR}/PCA-Deezer+MSD.csv"
DEEZER_SEG_100  = f"{DEEZER_DIR}/Deezer+Segments-100cnt.csv"
DEEZER_SEG_D30  = f"{DEEZER_DIR}/Deezer+Segments-030sec.csv"

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
DEF_DISTANCES   = algos.euclidean_score
DEF_NEIGHBORS_K = 7
DEF_SEGMENTS_CT = 100
DEF_SEGMENTS_DR = 30

ARG_SEGMENTS = [
    ("dur", 60), ("cnt", 200),
    ("dur", 50), ("cnt", 150),
    ("dur", 40), ("cnt", 100),
    ("dur", 30), ("cnt", 75),
    ("dur", 20), ("cnt", 50),
    ("dur", 10), ("cnt", 25),
    ("dur", 5), ("cnt", 10),
    ("dur", 2), ("cnt", 5),
    ("dur", 1), ("cnt", 1),
]

def LOAD_DATASETS(cols):
    arg_datasets = [
        SongDataset(
            name="Deezer",
            cols=cols["deezer"],
            path=DEEZER_SPO_MSD, knn=True, verbose=True,
            feat_index = 3, arousal = 4, valence = 3,
        ),
        SongDataset(
            name="Deezer+Spotify",
            cols=cols["deezer"] + cols["spotify"],
            path=DEEZER_SPO_MSD, knn=True, verbose=True,
        ),
        SongDataset(
            name="Deezer+MSD",
            cols=cols["deezer"] + cols["msd"],
            path=DEEZER_SPO_MSD, knn=True, verbose=True,
        ),
        SongDataset(
            name="Deezer+Spotify+MSD",
            cols=cols["deezer"] + cols["spotify"] + cols["msd"],
            path=DEEZER_SPO_MSD, knn=True, verbose=True,
        ),
        SongDataset(
            name="PCA-Deezer+Spotify",
            path=DEEZER_PCA_SPO, 
            knn=True, verbose=True,
        ),
        SongDataset(
            name="PCA-Deezer+MSD",
            path=DEEZER_PCA_MSD, 
            knn=True, verbose=True,
        ),
        SongDataset(
            name="PCA-Deezer+Spotify+MSD",
            path=DEEZER_PCA_ALL, 
            knn=True, verbose=True,
        ),
        SegmentDataset(
            name="Deezer+Segments-100cnt",
            cols=cols["deezer"] + cols["segments"],
            path=DEEZER_SEG_100, knn=True, verbose=True,
        ),
        SegmentDataset(
            name="Deezer+Segments-030sec",
            cols=cols["deezer"] + cols["segments"],
            path=DEEZER_SEG_D30, knn=True, verbose=True,
        )
    ]
    return arg_datasets

def LOAD_SEGMENT_DATASETS(cols, deezer_dir = DEEZER_DIR, knn = True):
    datasets = []

    for mode, num in ARG_SEGMENTS:
        path = "{}/segments/{}{:03}.csv".format(deezer_dir, mode, num)
        d = SegmentDataset(
            name="{}{:03}".format(mode, num),
            cols=cols["deezer"] + cols["segments"],
            path=path, knn=knn, verbose=True,
        )
        datasets.append(d)
    
    return datasets


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
    steps = steps * np.sqrt(2)
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

    count, total = 0, 0
    for i in range(N):
        for j in range(i+1, N):
            if not np.isnan(R[i][j]):
                count += 1
                total += R[i][j]
    
    return total / count

    # Get average of the sum of the triangle.
    # trisum = (R.sum() - np.trace(R))
    # triavg = trisum / (N * (N-1))
    # return triavg

def feat_stepvar(playlistDF, dataset):
    song_ids = playlistDF["id-deezer"].tolist()
    features = dataset.get_feats(song_ids).to_numpy()

    # Calculate step sizes.
    steplist = np.empty(0)
    for i in range(1, features.shape[0]):
        a, b = features[i], features[i-1]
        norm = np.linalg.norm(a - b)
        steplist = np.append(steplist, norm / np.sqrt(features.shape[1]))
    
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
        desc = df.groupby(variable)[m].describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]).round(6)
        desc.to_csv("{}/{}-{}.csv".format(dirname, variable, m))
    for fm in FEAT_METRICS:
        m = fm["func"].__name__
        desc = df.groupby(variable)[m].describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]).round(6)
        desc.to_csv("{}/{}-{}.csv".format(dirname, variable, m))

def plot_scores(df, variable, dirname):
    for pm in POINT_METRICS:
        m = pm["func"].__name__
        plot.snsplot(sns.boxenplot, df, m, variable, file="{}/{}-{}.png".format(dirname, variable, m))
    for fm in FEAT_METRICS:
        m = fm["func"].__name__
        plot.snsplot(sns.boxenplot, df, m, variable, file="{}/{}-{}.png".format(dirname, variable, m))
