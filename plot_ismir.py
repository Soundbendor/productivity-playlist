# import necessary modules
import numpy as np
import random
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.mstats import winsorize
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

# Basic output stuff.
outdir = "out/ismir2023"
helper.makeDir(outdir)
metrics = ["feat_pearson", "feat_stepvar", "pearson", "stepvar", "meansqr"]
dates = {
    "dataset":  "23-04-09-1750",
    "kval":     "23-04-11-1049",
    "distance": "23-04-09-2119",
    # "length":   "23-04-14-0157"
}

# Plot Deezer again - bigger.
info = helper.loadConfig("config.json")

def plotRussell(df, name, alpha=0.5):
    # mms = MinMaxScaler(feature_range=(-1,1))
    # valence = mms.fit_transform(songdata.va_df[["valence"]])
    # arousal = mms.fit_transform(songdata.va_df[["arousal"]])
    plot.av_circle(
        df["valence"], df["arousal"], 
        file="{}/{}.png".format(outdir, name), alpha=alpha
    )

    return


def plotExPlaylist(dataset):
    # 3135555   = Daft Punk         - Digital Love          (0.950468991,0.572575336)
    # 3135561   = Daft Punk         - Something About Us    (-0.317973857,-0.399224044)
    # 540954    = Rachael Yamagata  - 1963                  (1.070081923,1.018911652)
    # 533164    = Patty Loveless    - How Can I Help U ...  (-1.636899729,-0.45914527)

    user_orig, user_dest = 533164, 540954

    # Plot an example playlist.
    playlistDF = prodplay.makePlaylist(
        songdata, user_orig, user_dest, 11, verbose = 0, score=algos.euclidean_score, neighbors=7
    )

    plot.playlist([playlistDF[["valence", "arousal"]].to_numpy()], 
        file = f"{outdir}/ex-playlist.png", scale=0.5, axislabels=False
    )

    playlistDF.round(2).to_latex(hrules=True,
        buf=f"{outdir}/ex-playlist.tex",
        columns=["artist", "title", "valence", "arousal"],
        index=False
    )

    return

def dist(df):
    # DISTANCES
    df = df.replace([
        "Cosine Similarity", "Euclidean Distance", "Manhattan Distance", 
        "Jaccard Distance", "Multiplied Ratios", "Random Neighbors"
    ], [
        "Cosine", "Euclidean", "Manhattan", "Jaccard", "Ratios", "Random"
    ])
    df = df[df.distance != "Ratios"]

    # ## Table for feature-based Pearson correlation.
    # distfp = df.groupby("distance")["feat_pearson"].describe().round(6)
    # # print(distfp)
    # distfp[["mean", "std"]].style.to_latex(hrules=True, buf=f"{outdir}/dist-audio-pearson.tex")

    ## Plot for feature-based step variance.
    plot.snsplot(
        sns.boxenplot, df, "feat_stepvar", "distance", 
        file=f"{outdir}/dist-audio-stepvar.png", 
        scale=0.4
    )

    return

def data(df):
    # DATASETS
    df = df.replace([
        "Deezer", 
        "Deezer+Spotify", "Deezer+MSD", "Deezer+Spotify+MSD", 
        "PCA-Deezer+Spotify", "PCA-Deezer+MSD", "PCA-Deezer+Spotify+MSD",
        "Deezer+Segments-100cnt", "Deezer+Segments-030sec"
    ],[
        "Deezer", 
        "Spotify", "MSD", "all", 
        "PCA-Spotify", "PCA-MSD", "PCA-all",
        "100cnt", "030sec"
    ])
    df = df[df.dataset != "100cnt"]
    df = df[df.dataset != "030sec"]

    datamp = df.groupby("dataset")["pearson"].describe().round(4)
    datafp = df.groupby("dataset")["feat_pearson"].describe().round(4)
    datasv = df.groupby("dataset")["feat_stepvar"].describe().round(4)

    ## Mood-Pearson - table.
    datamp[["mean", "std", "50%"]].style.to_latex(hrules=True, buf=f"{outdir}/data-mood-pearson.tex")    

    ## Mood-Pearson - boxplot.
    plot.snsplot(
        sns.boxenplot, df, "pearson", "dataset", 
        file=f"{outdir}/data-mood-pearson.png", 
        scale=0.4
    )

    ## Audio-based metrics - table.
    audiotable = pd.merge(
        datafp[["mean", "std"]], datasv[["mean", "std"]],
        left_index=True, right_index=True, suffixes=("-pearson", "-stepvar")
    )
    audiotable.style.to_latex(hrules=True, buf=f"{outdir}/data-audio.tex")


    ## Smoothness metrics - table.
    smoothtable = pd.merge(
        datafp[["mean", "std", "50%"]], datamp[["mean", "std", "50%"]],
        left_index=True, right_index=True, suffixes=("-audio", "-mood")
    )
    smoothtable.style.to_latex(hrules=True, buf=f"{outdir}/data-pearson.tex")

    ## Audio step variance - boxplot.
    plot.snsplot(
        sns.boxenplot, df, "feat_stepvar", "dataset", 
        file=f"{outdir}/data-audio-stepvar.png", 
        scale=0.4
    )

    return

def kval(df):
    ## K-value graph - line with some spread marker.
    for m in ["pearson", "stepvar"]:
        plot.snsplot(
            plot.mult_y, df, "kval", [m, f"feat_{m}"],
            file=f"{outdir}/kval-{m}.png",
            figheight=9.6, figwidth=12.8, scale=0.35
        )

    return

if __name__ == "__main__":
    # Load up data from specific tests.
    dfs = {t: pd.read_csv(f"analysis/{dates[t]}-{t}s/_results.csv") for t in dates}

    dist(dfs["distance"])
    data(dfs["dataset"])
    kval(dfs["kval"])

    # songdata = SongDataset(
    #     name="Deezer+Spotify+MSD",
    #     cols=info["cols"]["deezer"] + info["cols"]["spotify"] + info["cols"]["msd"],
    #     path=testing.DEEZER_SPO_MSD, knn=True, verbose=True,
    # )
    # plotRussell(songdata.va_df, "circle-deezer")

    # samples = {}
    # with open("quadrants.json") as f: samples = json.load(f)
    # allpoints = []
    # for c in testing.QUADRANT_CODES: 
    #     for s in samples[c]:
    #         allpoints.append(int(s))
    # sampledata = songdata.get_point(allpoints)
    # plotRussell(sampledata, "circle-sample", alpha=1)




    









