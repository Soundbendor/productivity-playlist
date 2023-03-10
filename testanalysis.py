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
allresults["qc"] = allresults["oq"] + allresults["dq"]

# allresults = allresults[(allresults["qc"] != "TRBR") & (allresults["qc"] != "BRTR")]

testing.plot_scores(allresults, variable, analysisdir)
testing.metric_sheets(allresults, variable, analysisdir)
testing.plot_scores(allresults, "qc", analysisdir)
testing.metric_sheets(allresults, "qc", analysisdir)

for oq, dq, in testing.QUADRANT_COMBOS:
    qc = f"{oq}{dq}"
    print(f"- {qc}")

    quaddir = f"{analysisdir}/{qc}"
    helper.makeDir(quaddir)

    quadresults = pd.read_csv(f"{testdir}/{qc}/results-{samplecount}.csv", header=0, index_col=0)
    quadresults["qc"] = quadresults["oq"] + quadresults["dq"]
    
    testing.plot_scores(quadresults, variable, quaddir)
    testing.metric_sheets(quadresults, variable, quaddir)

    pointcombos = os.listdir(f"{testdir}/{qc}")
    pointcombos.remove(f"results-{samplecount}.csv")

    helper.makeDir(f"{analysisdir}/{qc}/playlists")
    for pc in pointcombos:
        playlistsDir = f"{testdir}/{qc}/{pc}"
        orig, dest = pc.split("-")
        
        legend = [name.split(".csv")[0] for name in os.listdir(playlistsDir)]
        playlistDFs = [pd.read_csv(f"{playlistsDir}/{name}.csv") for name in legend]
        testpoints = [df[["valence", "arousal"]].to_numpy() for df in playlistDFs]
        
        plot.playlist(testpoints, legend=legend,
            file = f"{analysisdir}/{qc}/playlists/{pc}.png",
            title = "Playlist from {} to {} based on {}".format(orig, dest, variable)
        )



