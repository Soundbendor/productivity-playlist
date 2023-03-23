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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer, KBinsDiscretizer

#our modules
import helper
import prodplay
import spotify
import plot
import algos
import testing
from analyze_data import discretize, analyze_dataset
from songdataset import SongDataset, SegmentDataset

info = helper.loadConfig("config.json")
indir = "./data/deezer"
outdir = f"{indir}/powert"
helper.makeDir(outdir)
datasets = testing.load_segment_datasets(
    info["cols"], "./data/deezer", knn=False)

# datasets = [
#     SongDataset(
#         name="Deezer+Spotify+MSD",
#         cols=info["cols"]["deezer"] + info["cols"]["spotify"] + info["cols"]["msd"],
#         path=testing.DEEZER_SPO_MSD, knn=True, verbose=True,
#         feat_index = 5, arousal = 4, valence = 3,
#     ),
#     SongDataset(
#         name="PCA-Deezer+Spotify",
#         path=testing.DEEZER_PCA_SPO, 
#         knn=True, verbose=True,
#         feat_index = 5, arousal = 4, valence = 3,
#     ),
#     SongDataset(
#         name="PCA-Deezer+MSD",
#         path=testing.DEEZER_PCA_MSD, 
#         knn=True, verbose=True,
#         feat_index = 5, arousal = 4, valence = 3,
#     ),
#     SongDataset(
#         name="PCA-Deezer+Spotify+MSD",
#         path=testing.DEEZER_PCA_ALL, 
#         knn=True, verbose=True,
#         feat_index = 5, arousal = 4, valence = 3,
#     ),
#     SegmentDataset(
#         name="Deezer+Segments-100cnt",
#         cols=info["cols"]["deezer"] + info["cols"]["segments"],
#         path=testing.DEEZER_SEG_100, knn=True, verbose=True,
#         feat_index = 5, arousal = 4, valence = 3,
#     ),
#     SegmentDataset(
#         name="Deezer+Segments-030sec",
#         cols=info["cols"]["deezer"] + info["cols"]["segments"],
#         path=testing.DEEZER_SEG_D30, knn=True, verbose=True,
#         feat_index = 5, arousal = 4, valence = 3,
#     )
# ]

def scaledata(df, cols, sc):
    ## Scale all the columns to the specific scaler.
    print(f"\nIndividually scaling using {sc.fit_transform.__name__}")
    for col in cols:
        print("... {}".format(col))
        df[[col]] = sc.fit_transform(df[[col]])
    return df

discretes = {}
for dataset in datasets:
    print(f"Transforming {dataset.name}")
    analyze_dataset(dataset, f"data/_analysis/{dataset.name}")

    df_scale = dataset.full_df.copy()
    cols = pd.merge(dataset.va_df, dataset.feat_df, left_index=True, right_index=True).columns

    scaledata(df_scale, cols, PowerTransformer(method='yeo-johnson', standardize=True))
    discretes[dataset.name] = discretize(df_scale, cols)
    scaledata(df_scale, cols, MinMaxScaler(feature_range=(-1,1)))
    df_scale.to_csv(f"{outdir}/{dataset.name}.csv")

    scaled_dataset = SongDataset(
        name=f'scaled-{dataset.name}',
        cols=dataset.cols,
        path=f"{outdir}/{dataset.name}.csv",
        feat_index = dataset.feat_index,
    )
    analyze_dataset(
        scaled_dataset, f"data/_analysis/{dataset.name}/scaled/powert")

helper.jsonout(discretes, "out/discretes-segments.json")




