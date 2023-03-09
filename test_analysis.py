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

variable = sys.argv[1] if len(sys.argv) > 1 else "dataset"
testtime = sys.argv[2] if len(sys.argv) > 2 else "23-03-09-1122"
samplecount = sys.argv[3] if len(sys.argv) > 3 else 5
testdir = f"test/{testtime}-{variable}s"

helper.makeDir("analysis")
analysisdir = f"./analysis/{testtime}-{variable}s"
helper.makeDir(analysisdir)

print(f"Analyzing {variable}s from {testtime}")
allresults = pd.read_csv(f"{testdir}/all-{samplecount}.csv", header=0, index_col=0)
testing.plot_scores(allresults, variable, analysisdir)

for oq, dq, in testing.QUADRANT_COMBOS:
    qc = f"{oq}{dq}"
    print(f"- {qc}")

    quaddir = f"{analysisdir}/{qc}"
    helper.makeDir(quaddir)

    quadresults = pd.read_csv(f"{testdir}/{qc}/results-{samplecount}.csv", header=0, index_col=0)
    testing.plot_scores(quadresults, variable, quaddir)
