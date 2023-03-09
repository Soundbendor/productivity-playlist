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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer


#our modules
import helper
import prodplay
import spotify
import plot
import algos
import testing
from songdataset import SongDataset
from segmentdataset import SegmentDataset

helper.makeDir("data/_analysis")
info = helper.loadConfig("config.json")
datasets = [
    # SongDataset(
    #     name="Deezer",
    #     cols=info["cols"]["deezer"],
    #     path=testing.DEEZER_STD_ALL,
    #     feat_index = 3
    # ),
    SongDataset(
        name="Deezer+Spotify",
        cols=info["cols"]["deezer"] + info["cols"]["spotify"],
        path=testing.DEEZER_STD_ALL,
    ),
    # SongDataset(
    #     name="Deezer+MSD",
    #     cols=info["cols"]["deezer"] + info["cols"]["msd"],
    #     path=testing.DEEZER_STD_ALL,
    # ),
    # SongDataset(
    #     name="PCA-Deezer+Spotify",
    #     path=testing.DEEZER_PCA_SPO, 
    # ),
    # SongDataset(
    #     name="PCA-Deezer+MSD",
    #     path=testing.DEEZER_PCA_MSD, 
    # ),
    # SongDataset(
    #     name="PCA-Deezer+Spotify+MSD",
    #     path=testing.DEEZER_PCA_ALL, 
    # ),
    # SegmentDataset(
    #     name="Deezer+Segments-100cnt",
    #     cols=info["cols"]["deezer"] + info["cols"]["segments"],
    #     path=testing.DEEZER_SEG_100,
    # ),
    # SegmentDataset(
    #     name="Deezer+Segments-030sec",
    #     cols=info["cols"]["deezer"] + info["cols"]["segments"],
    #     path=testing.DEEZER_SEG_100,
    # )
]
scalers = [
    {"name": "stdscl", "func": StandardScaler()},
    {"name": "minmax", "func": MinMaxScaler(feature_range=(-1,1))},
    {"name": "robust", "func": RobustScaler(quantile_range=(25,75))},
    {"name": "qtunif", "func": QuantileTransformer(output_distribution='uniform')},
    {"name": "qtnorm", "func": QuantileTransformer(output_distribution='normal')},
    {"name": "powert", "func": PowerTransformer(method='yeo-johnson', standardize=False)}
]

def analyze_dataset(dataset, dirname):
    helper.makeDir(dirname)
    feats = dataset.feat_df
    va = dataset.va_df
    df = pd.merge(va, feats, left_index=True, right_index=True)
    print(df.info())

    ## Basic descriptive stats.
    desc_stats = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Mean': df.mean(),
        'Median': df.median(),
        'Mode': df.mode().iloc[0]
    })
    print(desc_stats)
    desc_stats.to_csv("{}/mmm.csv".format(dirname), float_format="%.6f")
    
    ## Full descriptive stats.
    description = df.describe([0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]).T
    print(description)
    description.to_csv("{}/description.csv".format(dirname), float_format="%.6f")

    df.boxplot(figsize = (len(df.columns) * 1.5,10))
    plt.tight_layout()
    plt.savefig("{}/all-boxes.png".format(dirname))
    plt.close()

    ## Plot bounding boxes of each feature.
    helper.makeDir(f"{dirname}/feats")
    print("\nMaking individual plots for")
    for col in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
        print("... {}".format(col))        

        sns.boxenplot(data=df[[col]], ax=ax1)
        ax1.set_title(f"Boxplot for {col}")
        sns.histplot(data=df[[col]], ax=ax2)
        ax2.set_title(f"Histogram for {col}")

        plt.savefig(f"{dirname}/feats/{col}.png")
        plt.close()

    # plt.figure(figsize=(len(df.columns), len(df.columns)))
    # sns.pairplot(df)
    # plt.savefig("{}/pairplot.png".format(dirname))
    # plt.close()

    ## Correlation matrix and heatmap.
    correlation = df.corr().round(2)
    correlation.to_csv("{}/correlation.csv".format(dirname), float_format="%.6f")
    plt.figure(figsize=(len(df.columns) * 0.5, len(df.columns) * 0.4))
    sns.heatmap(correlation, annot=True, center=0)
    plt.tight_layout()
    plt.savefig("{}/heatmap.png".format(dirname))
    plt.close()

    return df

for dataset in datasets:
    ## Grab features and point data and analyze.
    dirname = "data/_analysis/{}".format(dataset.name)
    df = analyze_dataset(dataset, dirname)

    for scaler in scalers:
        print(f'\n\nUsing {scaler["name"]} scaler')
        df_scale = dataset.full_df.copy()
        
        ## Scale all the columns to the specific scaler.
        print("\nIndividually scaling")
        for col in df.columns:
            print("... {}".format(col))
            df_scale[[col]] = scaler["func"].fit_transform(df[[col]])
        
        ## Output the scaled dataset to scaler folder.
        dirscaled = f'{dirname}/scaled/{scaler["name"]}'
        helper.makeDir(dirscaled)
        df_scale.to_csv(f'{dirscaled}/data.csv')

        scaled_dataset = SongDataset(
            name=f'{scaler["name"]}-{dataset.name}',
            cols=dataset.cols,
            path=f'{dirscaled}/data.csv',
            feat_index = dataset.feat_index,
        )

        analyze_dataset(scaled_dataset, dirscaled)
