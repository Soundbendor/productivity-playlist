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

root     = "./data/affsample"
ids      = [int(i[0:8]) for i in os.listdir("{}/frames".format(root))]
# framecnt = pd.read_csv("{}/analysis/summary.csv".format(root), index_col=0, header=0, usecols=[0,1])
allframe = pd.read_csv("{}/data.csv".format(root), header=0, index_col=0)

idsUsed  = []
sample   = {
    "video": [],
    "frame": [],
    "arousal": [],
    "valence": []
}

for i in range(0, len(ids), 2):
    fid = ids[i]
    idsUsed.append(fid)
    print(fid, end="\r")

    sample["video"].append(allframe.loc[fid]['video'])
    sample["frame"].append(allframe.loc[fid]['frame'])
    sample["arousal"].append(allframe.loc[fid]['arousal'])
    sample["valence"].append(allframe.loc[fid]['valence'])

    orig = "{}/frames/{}.png".format(root, fid).encode('unicode-escape')
    dest = "{}3/frames/{}.png".format(root, fid).encode('unicode-escape')
    shutil.copyfile(orig, dest)

pd.DataFrame(sample, index=idsUsed).to_csv(path_or_buf="data/affsample3/data.csv")

# stats = {}
# stats["count"] = len(idsUsed)
# for d in ["valence", "arousal"]:
#     stats[d] = {
#         "mean": np.nanmean(sample[d]),
#         "std": np.nanstd(sample[d]),
#         "var": np.nanvar(sample[d]),
#         "min": np.nanmin(sample[d]),
#         "max": np.nanmax(sample[d]),
#         "median": np.nanmedian(sample[d])        
#     }

# json_obj = json.dumps(stats, indent=4)
# with open("data/affwild/downsample-stats.json", "w") as outfile:
#     outfile.write(json_obj)

# helper.plot_AV_data(
#     sample["valence"], sample["arousal"],
#     title="Spread of AffWild Sample",
#     file="data/affwild/downsample-cir.png",
#     alpha=.05
# )

# helper.plot_AV_box(
#     [sample["valence"], sample["arousal"]], ["valence","arousal"],
#     title="Distribution of AffWild Points",
#     file="data/affwild/downsample-box.png",
# )

# hists = [(i, sample[i]) for i in ["valence", "arousal"]]
# for attr, arr in hists:
#     helper.graph(
#         xlabel=attr, ylabel="Count", data=arr, hist=True,
#         title="{} Distribution in Sample".format(attr), 
#         file="data/affwild/downsample-{}.png".format(attr)
#     )