# import necessary modules
import numpy as np
import random
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import json
from pprint import pprint
import time
import sys
import os
import math
import multiprocessing
import itertools

#our modules
import helper
import prodplay
import algos
import plot
import testing
from songdataset import SongDataset, SegmentDataset

def perQuadrant(oq, dq):
    qc = f"{oq}{dq}"
    quaddir = f"{analysisdir}/{featEvalDataset.name}/{qc}"
    helper.makeDir(quaddir)

    # collect table of results
    results = {}
    for col in resultcols: results[col] = []

    pointcombos = os.listdir(f"{testdir}/{qc}/{featEvalDataset.name}")
    if f"results-{samplecount}.csv" in pointcombos:
        pointcombos.remove(f"results-{samplecount}.csv")
    # helper.makeDir(f"{analysisdir}/{qc}/playlists")

    for idx, pc in enumerate(pointcombos):
        # print(f"{qc} ... {idx + 1} / {len(pointcombos)}\t")
        playlistsDir = f"{testdir}/{qc}/{featEvalDataset.name}/{pc}"
        orig, dest = pc.split("-")
        orig, dest = int(orig), int(dest)
        
        legend = [name.split(".csv")[0] for name in os.listdir(playlistsDir)]
        playlistDFs = [pd.read_csv(f"{playlistsDir}/{name}.csv") for name in legend]

        for i in range(len(legend)):
            playlistDF = playlistDFs[i]
            name = legend[i]

            # Add results to our collection
            results[variable].append(name)
            results["oq"].append(oq)
            results["dq"].append(dq)
            results["orig"].append(orig)
            results["dest"].append(dest)

            # Evaluate playlist with each metric
            evals = testing.evaluate(playlistDF, featEvalDataset)
            for key in evals:
                results[key].append(evals[key])

        # testpoints = [df[["valence", "arousal"]].to_numpy() for df in playlistDFs]
        
        # plot.playlist(testpoints, legend=legend,
        #     file = f"{analysisdir}/{qc}/playlists/{pc}.png",
        #     title = "Playlist from {} to {} based on {}".format(orig, dest, variable)
        # )

    # quadresults = pd.read_csv(f"{testdir}/{qc}/results-{samplecount}.csv", header=0, index_col=0)
    quadresults = pd.DataFrame(results)
    quadresults.to_csv(f"{quaddir}/_results.csv")
    # dfs.append(quadresults)

    testing.plot_scores(quadresults, variable, quaddir)
    testing.metric_sheets(quadresults, variable, quaddir)
    return quadresults 

def getMostRecentTest(variable):
    testnames = list(reversed(sorted(os.listdir("./test"))))
    for name in testnames:
        if variable in name:
            date = name.split(f'-{variable}')[0]
            return date
    
if __name__ == '__main__':
    info = helper.loadConfig("config.json")

    variable = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    testtime = sys.argv[2] if len(sys.argv) > 2 else getMostRecentTest(variable)
    samplecount = sys.argv[3] if len(sys.argv) > 3 else 100
    testdir = f"test/{testtime}-{variable}s"

    helper.makeDir("analysis")
    analysisdir = f"./analysis/{testtime}-{variable}s"
    helper.makeDir(analysisdir)
    print(f"Analyzing {variable}s from {testtime}")

    # allresults = None
    # if os.path.isfile(f"{analysisdir}/_results.csv"):
    #     allresults = pd.read_csv(f"{analysisdir}/_results.csv")
    # else:
    # featEvalDataset = SongDataset(
    #     name="Deezer+Spotify+MSD",
    #     cols=info["cols"]["deezer"] + info["cols"]["spotify"] + info["cols"]["msd"],
    #     path=testing.DEEZER_SPO_MSD, verbose=True,
    # )

    featEvalDatasets = [
        SongDataset(
            name="Spotify",
            cols=info["cols"]["deezer"] + info["cols"]["spotify"],
            path=testing.DEEZER_SPO_MSD, verbose=True,
        ),
        SongDataset(
            name="MSD",
            cols=info["cols"]["deezer"] + info["cols"]["msd"],
            path=testing.DEEZER_SPO_MSD, verbose=True,
        ),
        SongDataset(
            name="All",
            cols=info["cols"]["deezer"] + info["cols"]["spotify"] + info["cols"]["msd"],
            path=testing.DEEZER_SPO_MSD, verbose=True,
        ),
        SongDataset(
            name="PCA-Spotify",
            path=testing.DEEZER_PCA_SPO, 
            verbose=True,
        ),
        SongDataset(
            name="PCA-MSD",
            path=testing.DEEZER_PCA_MSD, 
            verbose=True,
        ),
        SongDataset(
            name="PCA-All",
            path=testing.DEEZER_PCA_ALL, 
            verbose=True,
        )
    ]

    # Columns for our result sheets
    resultcols = [variable, "oq", "dq", "orig", "dest"]
    for pm in testing.POINT_METRICS:
        resultcols.append(pm["func"].__name__)
    for fm in testing.FEAT_METRICS:
        resultcols.append(fm["func"].__name__)

    for featEvalDataset in featEvalDatasets:
        helper.makeDir(f"{analysisdir}/{featEvalDataset.name}")
        # Run a process for each quadrant combo (12 in total).
        pQuadrants = multiprocessing.Pool(len(testing.QUADRANT_COMBOS))
        dfs = pQuadrants.starmap(perQuadrant, testing.QUADRANT_COMBOS)    

        allresults = pd.concat(dfs)
        allresults["qc"] = allresults["oq"] + allresults["dq"]
        allresults.to_csv(f"{analysisdir}/{featEvalDataset.name}/_results.csv")

        helper.makeDir(f"{analysisdir}/{featEvalDataset.name}/_all")
        testing.metric_sheets(allresults, variable, f"{analysisdir}/{featEvalDataset.name}/_all")
        testing.metric_sheets(allresults, "qc", f"{analysisdir}/{featEvalDataset.name}/_all")

        allresults["meansqr"] = winsorize(allresults["meansqr"], limits=[0.01, 0.05])
        allresults.reset_index(drop = True, inplace = True)

        testing.plot_scores(allresults, variable, f"{analysisdir}/{featEvalDataset.name}/_all")
        testing.plot_scores(allresults, "qc", f"{analysisdir}/{featEvalDataset.name}/_all")

    metrics = ["pearson","stepvar","meansqr","feat_pearson","feat_stepvar"]
    correlation = allresults[metrics].corr().round(2)
    correlation.to_csv("{}/correlation.csv".format(analysisdir), float_format="%.6f")
    plt.figure(figsize=(2 + len(metrics) * 0.6, 2 + len(metrics) * 0.4))
    sns.heatmap(correlation, annot=True, center=0, cmap='RdYlGn')
    plt.tight_layout()
    plt.savefig("{}/heatmap.png".format(analysisdir))
    plt.close()

    # winsresults = allresults.copy()


    # # someresults = allresults[(allresults["qc"] != "TRBR") & (allresults["qc"] != "BRTR") & (allresults["qc"] != "TLBL") & (allresults["qc"] != "BLTL")]

    # helper.makeDir(f"{analysisdir}/_wins")
    # testing.plot_scores(winsresults, variable, f"{analysisdir}/_wins")
    # testing.metric_sheets(winsresults, variable, f"{analysisdir}/_wins")
    # testing.plot_scores(winsresults, "qc", f"{analysisdir}/_wins")
    # testing.metric_sheets(winsresults, "qc", f"{analysisdir}/_wins")


