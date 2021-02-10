import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
import sys
import os
import helper
import shutil
import matplotlib as mpl

root     = "./data/affwild"
ids      = [int(i) for i in os.listdir("{}/frames".format(root))]
framecnt = pd.read_csv("{}/analysis/summary.csv".format(root), index_col=0, header=0, usecols=[0,1])
allframe = pd.read_csv("{}/analysis/all.csv".format(root), header=0, index_col=0)

idsUsed  = []
sample   = {
    "video": [],
    "frame": [],
    "arousal": [],
    "valence": []
}

for idx in ids:
    for frm in range(0, framecnt.loc[idx][0], 30):
        fid = int(1e5 * idx + frm)
        idsUsed.append(fid)
        print(fid, end="\r")

        sample["video"].append(idx)
        sample["frame"].append(frm)
        sample["arousal"].append(allframe.loc[fid]['arousal'])
        sample["valence"].append(allframe.loc[fid]['valence'])

        # orig = "{}/frames/{}/{}.png".format(root, idx, frm).encode('unicode-escape')
        # dest = "{}/downsample/{}.png".format(root, fid).encode('unicode-escape')
        # shutil.copyfile(orig, dest)

# pd.DataFrame(sample, index=idsUsed).to_csv(path_or_buf="data/affwild/downsample.csv")

stats = {}
stats["count"] = len(idsUsed)
for d in ["valence", "arousal"]:
    stats[d] = {
        "mean": np.nanmean(sample[d]),
        "std": np.nanstd(sample[d]),
        "var": np.nanvar(sample[d]),
        "min": np.nanmin(sample[d]),
        "max": np.nanmax(sample[d]),
        "median": np.nanmedian(sample[d])        
    }

json_obj = json.dumps(stats, indent=4)
with open("data/affwild/downsample-stats.json", "w") as outfile:
    outfile.write(json_obj)

helper.plot_AV_data(
    sample["valence"], sample["arousal"],
    title="Spread of AffWild Sample",
    file="data/affwild/downsample-cir.png",
    alpha=.05
)

helper.plot_AV_box(
    [sample["valence"], sample["arousal"]], ["valence","arousal"],
    title="Distribution of AffWild Points",
    file="data/affwild/downsample-box.png",
)

hists = [(i, sample[i]) for i in ["valence", "arousal"]]
for attr, arr in hists:
    helper.graph(
        xlabel=attr, ylabel="Count", data=arr, hist=True,
        title="{} Distribution in Sample".format(attr), 
        file="data/affwild/downsample-{}.png".format(attr)
    )