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
import testing
from songdataset import SongDataset, SegmentDataset

info = helper.loadConfig("config.json")
old_deezer_path = "data/deezer/original-info/all.csv"

deezer = SongDataset(
    name="Deezer",
    cols=["dzr_sng_id", "valence","arousal"],
    path=old_deezer_path, knn=True, verbose=True,
    feat_index = 0, arousal = 1, valence = 0,
)

scaled = SongDataset(
    name="Scaled",
    cols=info["cols"]["deezer"],
    path=testing.DEEZER_SPO_MSD, knn=True, verbose=True,
    feat_index = 3, arousal = 4, valence = 3,
)

for dataset in [deezer]:
    ## Unique point frequency table.
    freqs = [len(dataset.points_hash[helper.arr2stringPoint(u)]) for u in dataset.unique_points]
    v, a = np.transpose(dataset.unique_points)

    freqdf = pd.DataFrame({
        "valence": v,
        "arousal": a,
        "frequency": freqs
    })
    freqdf.sort_values(by="frequency", ascending=False, inplace=True)
    
    allstats = dataset.va_df.describe()
    print(allstats)

    freqstats = freqdf.describe()
    print(freqstats)

    jointstats = pd.merge(
        allstats, freqstats, 
        right_index=True, left_index=True, 
        suffixes=("-songs", "-points")
    )
    jointstats.style.to_latex(hrules=True, buf=f"out/desc-deezer.tex")

    # top20 = freqdf.head(n=20)
    # top20.to_latex(buf=f"out/top20duplicates.tex", index=False)
    # print(freqdf.head(n=20))

    # Distribution numbers.
    # Put the points in the quadrants.
    quadrant_names = ["BL", "BR", "TL", "TR"]
    quadrant_points = { "BL": 0, "BR": 0, "TL": 0, "TR": 0 }
    quadrant_songs = { "BL": 0, "BR": 0, "TL": 0, "TR": 0 }
    
    for point in dataset.unique_points:
        v, a = point[0], point[1]
        index = 0
        if v >= 0: index += 1
        if a >= 0: index += 2
        quadrant_points[quadrant_names[index]] += 1
        quadrant_songs[quadrant_names[index]] += len(
            dataset.points_hash[helper.arr2stringPoint(point)])
    


    # Print sizes out for sanity check.
    print("\nTotal points:", dataset.unique_size)
    print("Quadrants:\n{}\t{}\n{}\t{}".format(
        quadrant_points["TL"], quadrant_points["TR"], 
        quadrant_points["BL"], quadrant_points["BR"]
    ))

        # Print sizes out for sanity check.
    print("\nTotal songs:", len(dataset))
    print("Quadrants:\n{}\t{}\n{}\t{}".format(
        quadrant_songs["TL"], quadrant_songs["TR"], 
        quadrant_songs["BL"], quadrant_songs["BR"]
    ))


