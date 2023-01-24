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