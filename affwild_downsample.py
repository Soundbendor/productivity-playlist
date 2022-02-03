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

def make_dataset(name, ids):
    dest    = "./data/{}/".format(name)
    helper.makeDir(dest)
    helper.makeDir("{}/frames".format(dest))

    idsUsed  = []
    sample   = {
        "video": [],
        "frame": [],
        "arousal": [],
        "valence": []
    }

    for idx in ids:
        for frm in range(0, framecnt.loc[idx][0], 20):
            fid = int(1e5 * idx + frm)
            idsUsed.append(fid)
            print(fid, end="\r")

            sample["video"].append(idx)
            sample["frame"].append(frm)
            sample["arousal"].append(allframe.loc[fid]['arousal'])
            sample["valence"].append(allframe.loc[fid]['valence'])

            oldfile = "{}/frames/{}/{}.png".format(root, idx, frm).encode('unicode-escape')
            newfile = "{}/frames/{}.png".format(dest, fid).encode('unicode-escape')
            shutil.copyfile(oldfile, newfile)

    pd.DataFrame(sample, index=idsUsed).to_csv(path_or_buf="{}/data.csv".format(dest))

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
    with open("{}/stats.json".format(dest), "w") as outfile:
        outfile.write(json_obj)

    helper.plot_AV_data(
        sample["valence"], sample["arousal"],
        title="Spread of AffWild Sample",
        file="{}/cir.png".format(dest),
        alpha=.05
    )

    helper.plot_AV_box(
        [sample["valence"], sample["arousal"]], ["valence","arousal"],
        title="Distribution of AffWild Points",
        file="{}/box.png".format(dest),
    )

    hists = [(i, sample[i]) for i in ["valence", "arousal"]]
    for attr, arr in hists:
        helper.graph(
            xlabel=attr, ylabel="Count", data=arr, hist=True,
            title="{} Distribution in Sample".format(attr), 
            file="{}/{}.png".format(dest, attr)
        )


print(ids)

splitidx = int(len(ids) * 0.8)
ids_train   = ids[0:splitidx]
ids_test    = ids[splitidx:]
