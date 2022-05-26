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
import tests
from songdataset import SongDataset

# Make test directories if need be
all_test = "./ismir-test/"
run_time = sys.argv[1] if len(sys.argv) > 1 else "22-05-08_1207"
test_dir = "{}{}".format(all_test, run_time)
all_graph = "./ismir-analysis/"
helper.makeDir(all_graph)
graph_dir = "{}{}".format(all_graph, run_time)
helper.makeDir(graph_dir)

# Grab quadrants and quadrant combos
quadrants = ["BL", "BR", "TL", "TR"]
quadrant_combos = list(itertools.permutations(quadrants, 2))
quadrant_dirs = ["{}{}".format(a, b) for a, b in quadrant_combos]

# Grab metrics and column info.
base_cols = ["oq", "dq", "orig", "dest"] 
score_cols = ["cos_smooth", "cos_even", "euc_smooth", "euc_even"]
cos_songs = ["cos_{}".format(i) for i in range(10)]
euc_songs = ["euc_{}".format(i) for i in range(10)]

# Grab dataset names and start summary object.
datasets = os.listdir(test_dir)
summary = {}

for dataset in datasets:
    results = pd.read_csv(
        "{}/{}/results.csv".format(test_dir, dataset), header=0, index_col=0)
    summary[dataset] = {} 

    helper.makeDir("{}/{}".format(graph_dir, dataset))
    print(dataset)

    '''
        Data to get (stats and distplots for each):
         [X] four metrics overall
            [X] distplots for each
         [X] four metrics per quadrant
            [ ] distplots for each
         [ ] lengths of playlists (less than 10, 10, or more than 10)
            [ ] proportion of them by metric and dataset
    '''

    # Get stats for four metrics overall.
    for score in score_cols:
        print(" - {}".format(score))
        summary[dataset][score] = {}

        summary[dataset][score]["avg"] = float(np.nanmean(results[score]))
        summary[dataset][score]["std"] = float(np.nanstd(results[score]))
        summary[dataset][score]["var"] = float(np.nanvar(results[score]))
        summary[dataset][score]["min"] = float(np.nanmin(results[score]))
        summary[dataset][score]["max"] = float(np.nanmax(results[score]))
        summary[dataset][score]["med"] = float(np.nanmedian(results[score]))

        helper.graph(
            xlabel=score, ylabel="Count", data=results[score], hist=True,
            title="{} for {}".format(score, dataset),
            file="{}/{}/{}.png".format(graph_dir, dataset, score)
        )

        summary[dataset][score]["quadrants"] = {}
        for i in range(12):
            oq, dq = quadrant_combos[i]
            name = quadrant_dirs[i]
            print("   - {}".format(name))

            result = results[(results["oq"] == oq) & (results["dq"] == dq)]
            summary[dataset][score]["quadrants"][name] = {}

            summary[dataset][score]["quadrants"][name]["avg"] = float(
                np.nanmean(result[score]))
            summary[dataset][score]["quadrants"][name]["std"] = float(
                np.nanstd(result[score]))
            summary[dataset][score]["quadrants"][name]["var"] = float(
                np.nanvar(result[score]))
            summary[dataset][score]["quadrants"][name]["min"] = float(
                np.nanmin(result[score]))
            summary[dataset][score]["quadrants"][name]["max"] = float(
                np.nanmax(result[score]))
            summary[dataset][score]["quadrants"][name]["med"] = float(
                np.nanmedian(result[score])) 
    
json_obj = json.dumps(summary, indent=4)
with open("{}/summary.json".format(graph_dir), "w") as f:
    f.write(json_obj)