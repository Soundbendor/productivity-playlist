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
    # "Deezer+Segments-030sec", "Deezer+Segments-100cnt"
]
dates = {
    # "dataset":  "23-04-09-1750",
    # "kval":     "23-05-01-0005",
    "distance": "23-04-30-1302",
    # "length":   "23-04-30-1954",
    # "segment":  "23-04-30-0115"
}

# Plot Deezer again - bigger.
info = helper.loadConfig("config.json")

def plotRussell(df, name, alpha=0.5, quad=False):
    # mms = MinMaxScaler(feature_range=(-1,1))
    # valence = mms.fit_transform(songdata.va_df[["valence"]])
    # arousal = mms.fit_transform(songdata.va_df[["arousal"]])
    plot.av_circle(
        df["valence"], df["arousal"], 
        file="{}/{}.png".format(outdir, name), alpha=alpha, quad=quad
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
            "Spotify", "MSD", "All", 
            "PCA-Spotify", "PCA-MSD", "PCA-All",
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
    fig.set_figheight(4)

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
    fig.set_figheight(4)
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

    
    # fig, axs = plt.subplots(2, 2, dpi=600, sharex=True, sharey=True)
    # fig.set_figwidth(6)
    # fig.set_figheight(7)
    # plot.mult_y(ax=ax1, data=df, x="kval", y=["pearson", "feat_pearson"])
    # plot.mult_y(ax=ax2, data=df[df["qc"] == "BLTR"], x="kval", y=["stepvar", "feat_stepvar"])
    # # ax2.set_xlabel(None)
    # plt.tight_layout()

    # plt.savefig(f"{outdir}/kval.png", dpi=600)
    # plt.clf()
    # plt.close()
    
    return

def quadrants(results, deezer):

    def dir(oq, dq):
        if oq[1] == dq[1]: return "Vertical"
        elif oq[0] == dq[0]: return "Horizontal"
        else: return "Diagonal"
        return

    quads = ["BL", "BR", "TL", "TR"]
    qp = { "BL": 0, "BR": 0, "TL": 0, "TR": 0 }
    qs = { "BL": 0, "BR": 0, "TL": 0, "TR": 0 }
    
    for point in deezer.unique_points:
        v, a = point[0], point[1]
        index = 0
        if v >= 0: index += 1
        if a >= 0: index += 2
        qp[quads[index]] += 1
        qs[quads[index]] += len(
            deezer.points_hash[helper.arr2stringPoint(point)])

    tl = { "TR": "I", "TL": "II", "BL": "III", "BR": "IV" }
    qctrans = {
        f"{oq}{dq}": f"{tl[oq]} -> {tl[dq]}" for oq, dq in testing.QUADRANT_COMBOS
    }
    directions = {
        f"{oq}{dq}": dir(oq, dq) for oq, dq in testing.QUADRANT_COMBOS
    }

    pearson = results.groupby("qc")["pearson"].describe()
    stepvar = results.groupby("qc")["stepvar"].describe()

    # results["points"] = [qp[oq] + qp[dq] for oq, dq in zip(results["oq"], results["dq"])]
    # results["songs"] = 

    # quadinfo = pd.DataFrame([{
    #     "from": tl[oq], "to": tl[dq], "direction": directions[f"{oq}{dq}"],
    #     "songs": qs[oq] + qs[dq], "points": qp[oq] + qp[dq],
    # } for oq, dq in list(itertools.combinations(testing.QUADRANT_CODES, 2)) ])
    # print(quadinfo)
    # quadinfo.to_latex(buf="out/thesis-plots/quadinfo.tex", index=False)

    quadresults = pd.DataFrame([{
        "from": tl[oq], "to": tl[dq], "direction": directions[f"{oq}{dq}"],
        # "songs": qs[oq] + qs[dq],
        "points": qp[oq] + qp[dq],
        "pearson_avg": pearson.loc[f"{oq}{dq}", "mean"],
        "pearson_std": pearson.loc[f"{oq}{dq}", "std"],
        "pearson_med": pearson.loc[f"{oq}{dq}", "50%"],
        "stepvar_avg": stepvar.loc[f"{oq}{dq}", "mean"],
        "stepvar_std": stepvar.loc[f"{oq}{dq}", "std"],
        "stepvar_med": stepvar.loc[f"{oq}{dq}", "50%"],
    } for oq, dq in testing.QUADRANT_COMBOS ])

    quadresults = quadresults.sort_values(by="to", ascending=True)    
    quadresults = quadresults.sort_values(by="from", ascending=True)
    quadresults.to_latex(
        buf="out/thesis-plots/quadresults.tex", index=False,
        columns=[
            "from", "to", "direction", "points", 
            "pearson_avg", "pearson_std", "pearson_med", 
            "stepvar_avg", "stepvar_std", "stepvar_med",
        ]
    )
    print(f"len: {len(results)}\n")
    print(quadresults)

    # plot.snsplot(
    #     sns.lineplot, results, "points", "stepvar",
    #     file=f"{outdir}/quad-stepvar.png",
    #     figheight=4, figwidth=6, scale=1
    # )    

    return

def evals(test):
    filters = {
        "length": 11,
        "qc": "BLTR"
    }

    df = pd.read_csv(f"analysis/{test}/_results.csv")
    for f in filters:
        df = df[df[f] == filters[f]]
    print(df.info())

    examples = {
        "pearson": [0.3, 0.6,0.99],
        "stepvar": [0.0007, 0.0004, 0.0001],
    }

    colors = ['r', 'b', 'g']

    for m in examples:
        helper.makeDir(f"{outdir}/{m}")
        print(f"\n\n{m}:")

        fig, axs = plt.subplots(1, len(examples[m]), dpi=600)
        fig.set_figheight(2.4)
        fig.set_figwidth(6)

        for i, v in enumerate(examples[m]):
            df_closest = df.iloc[(df[m] - v).abs().argsort()[:1]]
            print(df_closest)
            orig = int(df_closest["orig"])
            dest = int(df_closest["dest"])
            print(orig, dest, v)

            pl = pd.read_csv(
                f"test/{test}/{filters['qc']}/{orig}-{dest}/{filters['length']}.csv"
            )

            axs[i].plot(
                [pl.iloc[0]["valence"], pl.iloc[-1]["valence"]],
                [pl.iloc[0]["arousal"], pl.iloc[-1]["arousal"]],
                marker=None, linestyle='dotted', color=colors[i]                
            )

            axs[i].plot(pl["valence"], pl["arousal"], marker='.', color=colors[i], linestyle='-')

            axs[i].set_frame_on(False)
            # axs[i].spines["bottom"].set_visible(False)

            axs[i].set_xlabel(f"{v}", fontweight='bold', size='large')
            axs[i].set_xticks([])
            axs[i].set_ylabel(None)
            axs[i].set_yticks([])

            # # print(pl)
            # plot.playlist(
            #     [pl[["valence", "arousal"]].to_numpy()], file=f"{outdir}/{m}/{v}.png"
            # )
        
        plt.tight_layout()
        plt.savefig(f"{outdir}/comp-{m}.png")
        plt.clf()
        plt.close()

    return

if __name__ == "__main__":
    # Load up data from specific tests.
    dfs = {t: {
        d: pd.read_csv(f"analysis/{dates[t]}-{t}s/{d}/_results.csv") for d in evaldata
     } for t in dates}

    songdata = SongDataset(
        name="Deezer",
        cols=info["cols"]["deezer"],
        path=testing.DEEZER_SPO_MSD, knn=True, verbose=True,
    )

    dist(dfs["distance"])
    # data(dfs["dataset"])
    # kval(dfs["kval"]["All"])
    # length(dfs["length"])
    # segments(dfs["segment"])

    # quadrants(dfs["dataset"]["All"], songdata)
    # plotRussell(songdata.va_df, "circle-deezer")

    # evals("23-04-09-1831-lengths")

    # samples = {}
    # with open("quadrants.json") as f: samples = json.load(f)
    # allpoints = []
    # for c in testing.QUADRANT_CODES: 
    #     for s in samples[c]:
    #         allpoints.append(int(s))
    # sampledata = songdata.get_point(allpoints)
    # plotRussell(sampledata, "circle-sample", alpha=1, quad=True)




    









