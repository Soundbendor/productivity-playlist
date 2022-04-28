import pandas as pd
import numpy as np
import helper
import json

songdata = pd.read_csv("./msdeezerplus.csv", header=0, index_col=0)

dirname = "./deezer-analysis"
helper.makeDir(dirname)
helper.makeDir("{}/charts".format(dirname))

summary = {}

for (name, data) in songdata.iteritems():
    coltype = str(songdata[name].dtype)
    isid = name[-2:] == 'id'
    print(name, coltype, isid)

    if isid or (coltype != 'int64' and coltype != 'float64'):
        print("not processing {} of type {}".format(name, coltype))
        continue

    summary[name] = {}
    summary[name] = {}
    summary[name]["mean"] = float(np.nanmean(data))
    summary[name]["std"] = float(np.nanstd(data))
    summary[name]["var"] = float(np.nanvar(data))
    summary[name]["min"] = float(np.nanmin(data))
    summary[name]["max"] = float(np.nanmax(data))
    summary[name]["median"] = float(np.nanmedian(data))

    helper.graph(
        xlabel=name, ylabel="Count", data=data, hist=True,
        title="{} Distribution".format(name),
        file="{}/charts/{}.png".format(dirname, name)
    )

json_obj = json.dumps(summary, indent=4)
with open("{}/summary.json".format(dirname), "w") as f:
    f.write(json_obj)