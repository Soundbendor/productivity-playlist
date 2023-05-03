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

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

#our modules
import helper
import prodplay
import algos
import plot
import testing
from songdataset import SongDataset, SegmentDataset

# Basic output stuff.
outdir = "out/thesis-plots"
helper.makeDir(outdir)
metrics = ["feat_pearson", "feat_stepvar", "pearson", "stepvar", "meansqr"]
evaldata = [
    "All", "Spotify", "MSD", 
    "PCA-All", "PCA-Spotify", "PCA-MSD", 
    "Deezer+Segments-030sec", "Deezer+Segments-100cnt"
]
dates = {
    # "dataset":  "23-04-09-1750",
    # "kval":     "23-04-11-1049",
    # "distance": "23-04-30-1302",
    # "length":   "23-04-30-1954",
    "segment":  "23-04-30-0115"
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

def dist(dfs):
    # DISTANCES
    df = dfs["All"]
    sp = dfs["Spotify"]

    df = df.replace([
        "Cosine Similarity", "Euclidean Distance", "Manhattan Distance", 
        "Jaccard Distance", "Multiplied Ratios", "Random Neighbors"
    ], [
        "Cosine", "Euclidean", "Manhattan", "Jaccard", "Ratios", "Random"
    ])
    df = df[df.distance != "Ratios"]

    sp = sp.replace([
        "Cosine Similarity", "Euclidean Distance", "Manhattan Distance", 
        "Jaccard Distance", "Multiplied Ratios", "Random Neighbors"
    ], [
        "Cosine", "Euclidean", "Manhattan", "Jaccard", "Ratios", "Random"
    ])
    sp = sp[sp.distance != "Ratios"]

    distsv = df.groupby("distance")["stepvar"].describe()
    distpc = df.groupby("distance")["pearson"].describe()
    distms = df.groupby("distance")["meansqr"].describe()
    distfp = df.groupby("distance")["feat_pearson"].describe()
    distfs = df.groupby("distance")["feat_stepvar"].describe()

    ## Mood-based metrics - table.
    moodtable = pd.merge(
        distpc[["mean", "std", "50%"]], distsv[["mean", "std", "50%"]],
        left_index=True, right_index=True, suffixes=("-pearson", "-stepvar")
    )
    moodtable.style.to_latex(hrules=True, buf=f"{outdir}/dist-mood.tex")

    # ## Table for feature-based Pearson correlation.
    audiotable = pd.merge(
        df.groupby("distance")["feat_pearson"].describe()[["mean", "std", "50%"]], 
        sp.groupby("distance")["feat_pearson"].describe()[["mean", "std", "50%"]],
        left_index=True, right_index=True, suffixes=("-all", "-spot")
    )
    audiotable.style.to_latex(hrules=True, buf=f"{outdir}/dist-audio-pearson.tex")
    # distfp[["mean", "std", "50%"]].style.to_latex(hrules=True, buf=f"{outdir}/dist-audio-pearson.tex")


    ## Plot for feature-based step variance.
    plot.snsplot(
        sns.boxenplot, df, "distance", "feat_stepvar", 
        file=f"{outdir}/dist-audio-stepvar.png", 
        figheight=3, figwidth=6, scale=1
    )

    return

def data(dfs):
    # DATASETS
    for d in evaldata:
        dfs[d] = dfs[d].replace([
            "Deezer", 
            "Deezer+Spotify", "Deezer+MSD", "Deezer+Spotify+MSD", 
            "PCA-Deezer+Spotify", "PCA-Deezer+MSD", "PCA-Deezer+Spotify+MSD",
            "Deezer+Segments-100cnt", "Deezer+Segments-030sec"
        ],[
            "Deezer", 
            "Spotify", "MSD", "all", 
            "PCA-Spotify", "PCA-MSD", "PCA-all",
            "100 segments", "30 seconds"
        ])

    df = dfs["All"]
    spot = dfs["Spotify"]

    datasv = df.groupby("dataset")["stepvar"].describe()
    datapc = df.groupby("dataset")["pearson"].describe()
    datams = df.groupby("dataset")["meansqr"].describe()
    datafp = df.groupby("dataset")["feat_pearson"].describe()
    datafs = df.groupby("dataset")["feat_stepvar"].describe()

    ## Mood-based metrics - table.
    moodtable = pd.merge(
        datapc[["mean", "std", "50%"]], datasv[["mean", "std", "50%"]],
        left_index=True, right_index=True, suffixes=("-pearson", "-stepvar")
    )
    moodtable.style.to_latex(hrules=True, buf=f"{outdir}/data-mood.tex")

    ## Audio-based metrics - table.
    audiotable = pd.merge(
        datafp[["mean", "std", "50%"]], datafs[["mean", "std", "50%"]],
        left_index=True, right_index=True, suffixes=("-pearson", "-stepvar")
    )
    audiotable.style.to_latex(hrules=True, buf=f"{outdir}/data-audio.tex")

    # Spotify features - boxplot.
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=600, sharey=True)
    fig.set_figwidth(6)
    fig.set_figheight(3.5)

    sns.boxenplot(ax=ax1, data=spot, x="feat_pearson", y="dataset")
    sns.boxenplot(ax=ax2, data=spot, x="feat_stepvar", y="dataset")
    ax1.set_xlabel("Pearson Correlation")
    ax2.set_xlabel("Step Size Variance")
    ax1.set_ylabel("Stage 2 Dataset")
    ax2.set_ylabel(None)
    plt.tight_layout()
    plt.savefig(f"{outdir}/data-spot.png", dpi=600)
    plt.clf()
    plt.close()

    # Mood Pearson - all vs BLTR.
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=600, sharey=True)
    fig.set_figwidth(6)
    fig.set_figheight(3.5)
    sns.boxenplot(ax=ax1, data=df, x="pearson", y="dataset")
    sns.boxenplot(ax=ax2, data=df[df["qc"] == "BLTR"], x="pearson", y="dataset")
    ax1.set_xlabel("PCC (all Quadrant paths)")
    ax2.set_xlabel("PCC (Quadrant III to I)")
    ax1.set_ylabel("Stage 2 Dataset")
    ax2.set_ylabel(None)
    plt.tight_layout()

    plt.savefig(f"{outdir}/data-mood-quadrants.png", dpi=600)
    plt.clf()
    plt.close()

    return

def kval(df):
    # K-value graph - line with some spread marker.
    for m in ["pearson", "stepvar"]:
        plot.snsplot(
            plot.mult_y, df, "kval", [m, f"feat_{m}"],
            file=f"{outdir}/kval-{m}.png",
            figheight=3.5, figwidth=6, scale=1
        )

    fig, (ax1, ax2) = plt.subplots(2, dpi=600, sharex=True)
    fig.set_figwidth(6)
    fig.set_figheight(7)
    plot.mult_y(ax=ax1, data=df, x="kval", y=["pearson", "feat_pearson"])
    plot.mult_y(ax=ax2, data=df[df["qc"] == "BLTR"], x="kval", y=["stepvar", "feat_stepvar"])
    # ax2.set_xlabel(None)
    plt.tight_layout()

    plt.savefig(f"{outdir}/kval.png", dpi=600)
    plt.clf()
    plt.close()


    return

def length(dfs):
    df = dfs["All"]
    order = testing.ARG_LENGTHS
    # df = df[df["length"] != 3]

    for m in ["pearson"]:
        plot.snsplot(
            plot.mult_y, df, "length", [m, f"feat_{m}"],
            file=f"{outdir}/length-{m}.png",
            figheight=3.5, figwidth=6, scale=1
        )

    steptable = pd.merge(
        df.groupby("length")["stepvar"].describe()[["mean", "std", "50%"]], 
        df.groupby("length")["feat_stepvar"].describe()[["mean", "std", "50%"]],
        left_index=True, right_index=True, suffixes=("-mood", "-audio")
    )
    steptable.style.to_latex(hrules=True, buf=f"{outdir}/length-stepvar.tex")
    return

def segments(dfs):
    df = dfs['Deezer+Segments-030sec']

    dur = df[df['segment'].str.contains("dur")]
    dur["duration"] = dur["segment"].map(lambda x : int(x[3:]))

    cnt = df[df['segment'].str.contains("cnt")]
    cnt["number"] = cnt["segment"].map(lambda x : int(x[3:]))

    for m in ["pearson", "stepvar"]:
        plot.snsplot(
            plot.mult_y, dur, "duration", [m, f"feat_{m}"],
            file=f"{outdir}/seg-duration-{m}.png",
            figheight=3.5, figwidth=6, scale=1
        )
        plot.snsplot(
            plot.mult_y, cnt, "number", [m, f"feat_{m}"],
            file=f"{outdir}/seg-number-{m}.png",
            figheight=3.5, figwidth=6, scale=1
        )

    
    # # fig, axs = plt.subplots(2, 2, dpi=600, sharex=True, sharey=True)
    # # fig.set_figwidth(6)
    # # fig.set_figheight(7)
    # # plot.mult_y(ax=ax1, data=df, x="kval", y=["pearson", "feat_pearson"])
    # # plot.mult_y(ax=ax2, data=df[df["qc"] == "BLTR"], x="kval", y=["stepvar", "feat_stepvar"])
    # # # ax2.set_xlabel(None)
    # # plt.tight_layout()

    # plt.savefig(f"{outdir}/kval.png", dpi=600)
    # plt.clf()
    # plt.close()
    
    return

if __name__ == "__main__":
    # Load up data from specific tests.
    dfs = {t: {
        d: pd.read_csv(f"analysis/{dates[t]}-{t}s/{d}/_results.csv") for d in evaldata
     } for t in dates}

    # dist(dfs["distance"])
    # data(dfs["dataset"])
    # kval(pd.read_csv("analysis/23-04-11-1049-kvals/_results.csv"))
    # length(dfs["length"])
    segments(dfs["segment"])

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




    









