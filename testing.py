#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
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

#our modules
import helper
import prodplay
import spotify
import plot
import algos
from songdataset import SongDataset
from segmentdataset import SegmentDataset

def pearson(playlistDF):
    points = playlistDF[["valence", "arousal"]].to_numpy()
    pr = np.corrcoef(points, rowvar=False)
    return pr[1][0]

def stepvar(playlistDF):
    steps = playlistDF["evenness"].to_numpy()[1:]
    sv = np.var(steps)
    return sv

def meansquare(playlistDF):
    points = playlistDF[["valence", "arousal"]].to_numpy()
    target = points.copy()

    # Generate ideal playlist array using orig, dest, n.
    orig, dest = points[0], points[-1]
    stepcount = len(points) - 1

    return None