import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
import sys
import os
import helper
import matplotlib as mpl

mpl.rcParams.update({'figure.max_open_warning': 0})
annotations = "./data/affwild/annotations/train"
ids = [int(i[0:3]) for i in os.listdir("{}/{}".format(annotations, "arousal"))]
frame_ids = []

summary = {
    "frame_count":[],
    "arousal_mean":[],
    "arousal_std":[],
    "arousal_var":[],
    "arousal_median":[],
    "arousal_min":[],
    "arousal_max":[],
    "valence_mean":[],
    "valence_std":[],
    "valence_var":[],
    "valence_median":[],
    "valence_min":[],
    "valence_max":[],
}

all_data = {
    "video": [],
    "frame": [],
    "arousal": [],
    "valence": []
}

maxlen = 0
for idx in ids:
    for d in ["valence", "arousal"]:
        n = 0

        with open("{}/{}/{}.txt".format(annotations, d, idx)) as f:
            content = [float(i) for i in f.readlines()]
            all_data[d] = all_data[d] + content

            if (d == "arousal"): 
                n = len(content)
                summary["frame_count"].append(n)

            summary["{}_mean".format(d)].append(np.nanmean(content))
            summary["{}_std".format(d)].append(np.nanstd(content))
            summary["{}_var".format(d)].append(np.nanvar(content))
            summary["{}_min".format(d)].append(np.nanmin(content))
            summary["{}_max".format(d)].append(np.nanmax(content))
            summary["{}_median".format(d)].append(np.nanmedian(content))

        frame_ids = frame_ids + [int(1e5 * idx + i) for i in range(n)]
        all_data["video"] = all_data["video"] + [idx for i in range(n)]
        all_data["frame"] = all_data["frame"] + [i for i in range(n)]

allstats = {}
for d in ["valence", "arousal"]:
    allstats[d] = {
        "mean": np.nanmean(all_data[d]),
        "std": np.nanstd(all_data[d]),
        "var": np.nanvar(all_data[d]),
        "min": np.nanmin(all_data[d]),
        "max": np.nanmax(all_data[d]),
        "median": np.nanmedian(all_data[d])        
    }

helper.makeDir("data/affwild/analysis")
pd.DataFrame(summary, index=ids).to_csv(path_or_buf="data/affwild/analysis/summary.csv")
pd.DataFrame(all_data, index=frame_ids).to_csv(path_or_buf="data/affwild/analysis/all.csv")

json_obj = json.dumps(allstats, indent=4)
with open("data/affwild/analysis/allstats.json", "w") as outfile:
    outfile.write(json_obj)

helper.plot_AV_data(
    summary["valence_mean"], summary["arousal_mean"],
    title="Spread of AffWild (Averages Per Video)",
    file="data/affwild/analysis/cir_sum.png"
)

columns = list(summary)[1:]
helper.plot_AV_box(
    [summary[i] for i in columns], columns,
    title="Distribution of AffWild Data",
    file="data/affwild/analysis/box_sum.png",
    vert=False
)

helper.plot_AV_data(
    all_data["valence"], all_data["arousal"],
    title="Spread of AffWild (All Points)",
    file="data/affwild/analysis/cir_all.png",
    alpha=.005
)

helper.plot_AV_box(
    [all_data["valence"], all_data["arousal"]], ["valence","arousal"],
    title="Distribution of AffWild Points",
    file="data/affwild/analysis/box_all.png",
)

helper.makeDir("data/affwild/analysis/hist")
hists = [(i, all_data[i]) for i in ["valence", "arousal"]] + [(i, summary[i]) for i in list(summary)]
for attr, arr in hists:
    helper.graph(
        xlabel=attr, ylabel="Count", data=arr, hist=True,
        title="{} Distribution in AffWild".format(attr), 
        file="data/affwild/analysis/hist/{}.png".format(attr)
    )