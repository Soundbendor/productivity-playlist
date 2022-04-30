#import numpy and pandas (for data) and NearestNeighbors (for neighbor calculations)
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import json
import pprint
import time
import sys
import os
import math

#our modules
import helper
import prodplay
import algos
import tests
from songdataset import SongDataset

songdata = SongDataset(
    name="Deezer",
    path="data/deezer/deezer-spotify.csv",
    cols=[0,3,6,7], 
    start_index = 1
)
songdata.make_knn()

test_time = str(time.strftime("%y-%m-%d_%H%M"))
helper.makeDir("./quadrants")

scalers = [
    ('mms', MinMaxScaler(feature_range=(-1,1))),
    ('std', StandardScaler())
]

def make_scale_quadrants(id, scaler):
    dirname = "./quadrants/{}-{}".format(id, test_time)
    helper.makeDir(dirname)
    scaled_points = scaler.fit_transform(pd.DataFrame(songdata.unique_points).iloc[:, 0:2])
    scaled_to_og = {}
    print("\nUsing {} scaler, data in directory {}".format(id, dirname))

    # Get a way to correspond from our scaled points back to the original points.
    for i in range(songdata.unique_size):
        origin_point = songdata.unique_points[i]
        scaled_point = scaled_points[i]
        scaled_string = helper.arr2stringPoint(scaled_point)
        scaled_to_og[scaled_string] = origin_point

    # Quadrants array: (-v,-a), (+v,-a), (-v,+a), (+v,+a)
    quadrant_names = ["BL", "BR", "TL", "TR"]
    quadrants = { "BL": [], "BR": [], "TL": [], "TR": [] }

    # Put the points in the quadrants.
    for point in scaled_points:
        v, a = point[0], point[1]
        index = 0
        if v >= 0: index += 1
        if a >= 0: index += 2
        quadrants[quadrant_names[index]].append([v, a])

    # Print sizes out for sanity check.
    print("Total points:", songdata.unique_size)
    print("Quadrants:\n{}\t{}\n{}\t{}".format(
        len(quadrants["TL"]), len(quadrants["TR"]), 
        len(quadrants["BL"]), len(quadrants["BR"])
    ))

    # Grab random sample points from each quadrant.
    samples = { "BL": [], "BR": [], "TL": [], "TR": [] }
    for name in quadrant_names:
        samples[name] = random.sample(quadrants[name], 100)

    # Plot all of the samples.
    allsamples = np.vstack((samples["TL"], samples["TR"], samples["BL"], samples["BR"]))

    if id != 'mms':
        allsamples = MinMaxScaler(feature_range=(-1,1)).fit_transform(allsamples)

    allsamples = np.transpose(allsamples)

    helper.plot_AV_data(
        allsamples[0], allsamples[1],
        title="random sample using {}".format(id),
        file="{}/cir.png".format(dirname)
    )

    # Grab songs from sample points.
    samplesongs = { "BL": [], "BR": [], "TL": [], "TR": [] }
    for name in quadrant_names:
        for p in samples[name]:
            pstring = helper.arr2stringPoint(p)
            origpoint = scaled_to_og[pstring]
            songid = songdata.get_song(origpoint)
            samplesongs[name].append(str(songid))

    # Print the points into a JSON file.
    json_samples = json.dumps(samplesongs, indent=4)
    with open("{}/songs.json".format(dirname), "w") as f:
        f.write(json_samples)

if __name__ == "__main__":
    for id, scaler in scalers:
        make_scale_quadrants(id, scaler)