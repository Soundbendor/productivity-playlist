import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
import sys
import os
import helper
from sklearn.preprocessing import MinMaxScaler

# Designed to compare Arousal-Valence music datasets.
# Combines the following functionalities:
#   - plot distribution (scatterplot, box & whisker)
#   - find data summaries (min, max, avg, med, stdev, etc.)
#   - find intersections (based on title & artist)

info = helper.loadConfig()
helper.makeDir("analysis")

datasets = []
names = []
points = []
namesets = []
for obj in info["analysis"]:
    names.append(obj["name"])
    datasets.append(pd.read_csv(obj["csv"], header=0, index_col=0, usecols=obj["cols"]))
    with open(obj["coords"]) as f: points.append(json.load(f))

summary = {}
for i in range(len(names)):
    summary[names[i]] = {}
    summary[names[i]]["size"] = len(datasets[i])
    summary[names[i]]["intersection"] = {}

    coords = []
    for key in points[i].keys():
        coords.append(helper.string2arrPoint(key))
    coords = np.array(coords)
    series = np.transpose(coords)

    valence = series[0]
    summary[names[i]]["valence"] = {}
    summary[names[i]]["valence"]["mean"] = np.nanmean(valence)
    summary[names[i]]["valence"]["std"] = np.nanstd(valence)
    summary[names[i]]["valence"]["var"] = np.nanvar(valence)
    summary[names[i]]["valence"]["min"] = np.nanmin(valence)
    summary[names[i]]["valence"]["max"] = np.nanmax(valence)
    summary[names[i]]["valence"]["median"] = np.nanmedian(valence)

    arousal = series[1]
    summary[names[i]]["arousal"] = {}
    summary[names[i]]["arousal"]["mean"] = np.nanmean(arousal)
    summary[names[i]]["arousal"]["std"] = np.nanstd(arousal)
    summary[names[i]]["arousal"]["var"] = np.nanvar(arousal)
    summary[names[i]]["arousal"]["min"] = np.nanmin(arousal)
    summary[names[i]]["arousal"]["max"] = np.nanmax(arousal)
    summary[names[i]]["arousal"]["median"] = np.nanmedian(arousal)
    
    scaler = MinMaxScaler(feature_range=(-1,1)) 
    scaled_values = np.transpose(scaler.fit_transform(pd.DataFrame(coords).iloc[:,0:2]))
    helper.plot_AV_data(
        scaled_values[0], scaled_values[1],
        title="Spread of {} Dataset".format(names[i]),
        file="analysis/cir_{}.png".format(names[i])
    )
    helper.plot_AV_box(
        arousal, valence, 
        title="Distribution of {} Dataset".format(names[i]),
        file="analysis/box_{}.png".format(names[i])
    )

    tuples = []
    for j in range(len(datasets[i])):
        tuples.append((datasets[i].iloc[j, 2], datasets[i].iloc[j, 3]))
    namesets.append(set(tuples))

pprint.pprint(namesets)

for i in range(len(namesets) - 1):
    for j in range(i+1, len(namesets)):
        intersection = namesets[i] & namesets[j]

        json_obj = json.dumps(list(intersection), indent=2)
        with open("analysis/int_{}_{}.json".format(names[i], names[j]), "w") as f:
            f.write(json_obj)
        
        summary[names[i]]["intersection"][names[j]] = len(intersection)
        summary[names[j]]["intersection"][names[i]] = len(intersection)

json_obj = json.dumps(summary, indent=4)
with open("analysis/summary.json", "w") as f:
    f.write(json_obj)