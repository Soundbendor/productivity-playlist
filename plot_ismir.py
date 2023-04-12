# import necessary modules
import numpy as np
import random
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
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

# Plot Deezer again - bigger.
info = helper.loadConfig("config.json")

songdata = SongDataset(
    name="Deezer+Spotify+MSD",
    cols=info["cols"]["deezer"] + info["cols"]["spotify"] + info["cols"]["msd"],
    path=testing.DEEZER_SPO_MSD, knn=True, verbose=True,
)
mms = MinMaxScaler(feature_range=(-1,1))
valence = mms.fit_transform(songdata.va_df[["valence"]])
arousal = mms.fit_transform(songdata.va_df[["arousal"]])
plot.av_circle(
    valence, arousal, 
    title=f"Spread of Deezer 2018",
    file="{}/deezer-circle.png".format(outdir)
)

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

playlistDF.round(2).to_latex(
    buf=f"{outdir}/ex-playlist.tex",
    columns=["artist", "title", "valence", "arousal"],
    index=False
)


# Load up data from specific tests.
# dates = {
#     "dataset": "23-04-09-1750",
#     "kval": "23-04-11-1049",
#     "distance": "23-04-09-2119",
#     "length": "23-04-10-2222"
# }
# dfs = {t: pd.read_csv(f"analysis/{dates[t]}-{t}s/_results.csv") for t in dates}

# # DISTANCES
# dfs["distance"] = dfs["distance"].replace([
#     "Cosine Similarity", "Euclidean Distance", "Manhattan Distance", 
#     "Jaccard Distance", "Multiplied Ratios", "Random Neighbors"
# ], [
#     "Cosine", "Euclidean", "Manhattan", "Jaccard", "Ratios", "Random"
# ])
# dfs["distance"] = dfs["distance"][dfs["distance"].distance != "Ratios"]

# ## Table for feature-based Pearson correlation.
# distfp = dfs["distance"].groupby("distance")["feat_pearson"].describe().round(6)
# # print(distfp)
# distfp[["mean", "std"]].style.to_latex(f"{outdir}/dist-fp.tex")

# ## Plot for feature-based step variance.
# plot.snsplot(
#     sns.boxenplot, dfs["distance"], "feat_stepvar", "distance", 
#     file=f"{outdir}/dist-sv.png", 
#     scale=0.4
# )

# # DATASETS
# ## Better boxplot for datasets.

# dfs["dataset"] = dfs["dataset"].replace([
#     "Deezer", 
#     "Deezer+Spotify", "Deezer+MSD", "Deezer+Spotify+MSD", 
#     "PCA-Deezer+Spotify", "PCA-Deezer+MSD", "PCA-Deezer+Spotify+MSD",
#     "Deezer+Segments-100cnt", "Deezer+Segments-030sec"
# ],[
#     "Deezer", 
#     "Spotify", "MSD", "all", 
#     "PCA-Spotify", "PCA-MSD", "PCA-all",
#     "100cnt", "030sec"
# ])
# dfs["dataset"] = dfs["dataset"][dfs["dataset"].dataset != "100cnt"]
# dfs["dataset"] = dfs["dataset"][dfs["dataset"].dataset != "030sec"]

# ## Table for each with median, mean, std and IQR.

# ### K-VALUES
# # plot.snsplot(
# #     sns.lineplot, dfs["kval"], "kval", "pearson",
# #     file=f"{outdir}/kval-mp.png"
# # )

# # plot.snsplot(
# #     sns.lineplot, dfs["kval"], "kval", "stepvar",
# #     file=f"{outdir}/kval-ms.png"
# # )

# # plot.snsplot(
# #     sns.lineplot, dfs["kval"], "kval", "feat_pearson",
# #     file=f"{outdir}/kval-fp.png"
# # )

# # plot.snsplot(
# #     sns.lineplot, dfs["kval"], "kval", "feat_stepvar",
# #     file=f"{outdir}/kval-fs.png"
# # )

# ## K-value graph - line with some spread marker.

# # https://matplotlib.org/3.4.3/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
# def mult_y(ax, data, x, y, palette=None):
#     lc = len(y)
#     axes = [None for _ in range(lc)]
#     axes[0] = ax
#     colors = list(mcolors.BASE_COLORS)
#     linestyles = ['-', '--', ':', '-.']
    
#     for i in range(1, lc): axes[i] = ax.twinx()
#     for i in range(2, lc): axes[i].spines.right.set_position(("axes", 0.8 + (0.2 * i)))

#     handles = []
#     for i in range(lc): 
#         sns.lineplot(
#             ax=axes[i], data=data, x=x, y=y[i], 
#             color=colors[i], linestyle=linestyles[i]
#         )
#         axes[i].yaxis.label.set_color(colors[i])
#         axes[i].tick_params(axis='y', colors=colors[i])
#         handles.append(
#             mlines.Line2D([], [], color=colors[i], linestyle=linestyles[i], label=y[i]))

#     ax.legend(labels=y, handles=handles)

# for m in ["pearson", "stepvar"]:
#     plot.snsplot(
#         mult_y, dfs["kval"], "kval", [m, f"feat_{m}"],
#         file=f"{outdir}/kval-{m}.png",
#         figheight=9.6, figwidth=12.8
#     )










