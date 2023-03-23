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
from songdataset import SongDataset, SegmentDataset

def analyze_dataset(dataset, dirname, verbose=1):
    helper.makeDir(dirname)
    feats = dataset.feat_df
    va = dataset.va_df
    df = pd.merge(va, feats, left_index=True, right_index=True)
    if verbose >= 2: print(df.info())

    ## Basic descriptive stats.
    desc_stats = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Mean': df.mean(),
        'Median': df.median(),
        'Mode': df.mode().iloc[0]
    })
    if verbose >= 2: print(desc_stats)
    desc_stats.to_csv("{}/mmm.csv".format(dirname), float_format="%.6f")
    
    ## Full descriptive stats.
    description = df.describe([0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]).T
    if verbose >= 2: print(description)
    description.to_csv("{}/description.csv".format(dirname), float_format="%.6f")

    ## Full boxplot.
    df.boxplot(figsize = (len(df.columns) * 1.5,10))
    plt.tight_layout()
    plt.savefig("{}/all-boxes.png".format(dirname))
    plt.close()

    # Plot bounding boxes of each feature.
    helper.makeDir(f"{dirname}/feats")
    if verbose >= 1: print("\nMaking individual plots for")
    for col in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
        if verbose >= 1: print("... {}".format(col))        

        sns.boxenplot(data=df[[col]], ax=ax1)
        ax1.set_title(f"Boxplot for {col}")
        sns.histplot(data=df[[col]], ax=ax2)
        ax2.set_title(f"Histogram for {col}")

        plt.savefig(f"{dirname}/feats/{col}.png")
        plt.close()
    
    # Correlation matrix and heatmap.
    correlation = df.corr().round(2)
    correlation.to_csv("{}/correlation.csv".format(dirname), float_format="%.6f")
    plt.figure(figsize=(len(df.columns) * 0.5, len(df.columns) * 0.4))
    sns.heatmap(correlation, annot=True, center=0)
    plt.tight_layout()
    plt.savefig("{}/heatmap.png".format(dirname))
    plt.close()

    # # Arousal-Valence circle plot.
    # mms = MinMaxScaler(feature_range=(-1,1))
    # valence = mms.fit_transform(df[["valence"]])
    # arousal = mms.fit_transform(df[["arousal"]])
    # plot.av_circle(
    #     valence, arousal, 
    #     title=f"Spread of {dataset.name}",
    #     file="{}/circle.png".format(dirname)
    # )

    return df

def discretize(df, columns, maxcoef=5.5, verbose=1):
    badfeats = []
    catfeats = [
        "sp_time_sig", "sp_explicit", "sp_mode",
        "MSD_key", "MSD_mode", "MSD_time_signature"
    ]
    if verbose >= 2: print("Analyzing...")
    
    for col in columns: 
        if col in catfeats: continue
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        
        if verbose >= 2: 
            print(col)
            # print(df[col].describe([.01, .05, .1, .2, .25, .5, .75, .8, .9, .95, .99]))
            print(" - iqr:", iqr)

        goodmin = df[col].quantile(0.25) - (maxcoef * iqr)
        goodmax = df[col].quantile(0.75) + (maxcoef * iqr)
        realmin = df[col].quantile(0)
        realmax = df[col].quantile(1)
        iqrfmin = abs((df[col].quantile(.25) - realmin) / iqr)
        iqrfmax = abs((df[col].quantile(.75) - realmax) / iqr)

        if realmax > goodmax or realmin < goodmin:
            if verbose >= 2:
                print(" - Min: good -", goodmin, ", actual -", realmin, ', distcoef -', iqrfmin)
                print(" - Max: good -", goodmax, ", actual -", realmax, ', distcoef -', iqrfmax)
                print(" - Time to discretize!")
            badfeats.append(col)
        # elif verbose >= 2:
        #     print(" - All good here :)")

    kbd = KBinsDiscretizer(n_bins = 20, encode='ordinal', strategy='quantile')
    if verbose >= 1: print("\n\nDiscretizing:")
    
    for col in badfeats:
        if verbose >= 1: print(f"... {col}")
        df[[col]] = kbd.fit_transform(df[[col]])
    
    if verbose >= 1: print("\n")
    return badfeats    

if __name__ == "__main__":

    helper.makeDir("data/_analysis")
    info = helper.loadConfig("config.json")
    datasets = [
        # SongDataset(
        #     name="Deezer",
        #     cols=info["cols"]["deezer"],
        #     path=testing.DEEZER_SPO_MSD,
        #     feat_index = 3
        # ),
        SongDataset(
            name="Deezer+Spotify",
            cols=info["cols"]["deezer"] + info["cols"]["spotify"],
            path=testing.DEEZER_SPO_MSD,
        ),
        SongDataset(
            name="Deezer+MSD",
            cols=info["cols"]["deezer"] + info["cols"]["msd"],
            path=testing.DEEZER_SPO_MSD,
        ),
        SongDataset(
            name="PCA-Deezer+Spotify",
            path=testing.DEEZER_PCA_SPO, 
        ),
        SongDataset(
            name="PCA-Deezer+MSD",
            path=testing.DEEZER_PCA_MSD, 
        ),
        SongDataset(
            name="PCA-Deezer+Spotify+MSD",
            path=testing.DEEZER_PCA_ALL, 
        ),
        SegmentDataset(
            name="Deezer+Segments-100cnt",
            cols=info["cols"]["deezer"] + info["cols"]["segments"],
            path=testing.DEEZER_SEG_100,
        ),
        SegmentDataset(
            name="Deezer+Segments-030sec",
            cols=info["cols"]["deezer"] + info["cols"]["segments"],
            path=testing.DEEZER_SEG_D30,
        )
    ]

    mms = MinMaxScaler(feature_range=(-1,1))
    scalers = [
        {"name": "stdscl", "func": StandardScaler()},
        {"name": "minmax", "func": MinMaxScaler(feature_range=(-1,1))},
        {"name": "robust", "func": RobustScaler(quantile_range=(25,75))},
        {"name": "qtunif", "func": QuantileTransformer(output_distribution='uniform')},
        {"name": "qtnorm", "func": QuantileTransformer(output_distribution='normal')},
        {"name": "powert", "func": PowerTransformer(method='yeo-johnson', standardize=True)}
    ]

    for dataset in datasets:
        ## Grab features and point data and analyze.
        dirname = "data/_analysis/{}".format(dataset.name)
        df = analyze_dataset(dataset, dirname)

        discretes = {}
        for scaler in scalers:
            print(f'\n\nUsing {scaler["name"]} scaler')
            df_scale = dataset.full_df.copy()

            ## Scale all the columns to the specific scaler.
            print("\nIndividually scaling")
            for col in df.columns:
                print("... {}".format(col))
                df_scale[[col]] = scaler["func"].fit_transform(df[[col]])

            # Discretize outlier columns.
            discretes[scaler["name"]] = discretize(df_scale, df.columns)
            
            print("\nMin Max Scaling to (-1,1)")
            for col in df.columns:
                df_scale[[col]] = mms.fit_transform(df_scale[[col]])
            
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
        
        helper.jsonout(discretes, f"data/_analysis/{dataset.name}/discretes.json")