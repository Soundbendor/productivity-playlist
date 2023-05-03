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

pointdata = SongDataset(
    name="Deezer",
    cols=info["cols"]["deezer"],
    path=datasetpath, knn=True, verbose=True,
    feat_index = 3, arousal = 4, valence = 3,
)

