import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
import json
import numpy as np
import pprint
import re
import time
from sklearn.decomposition import PCA
from songdataset import SongDataset

def loadConfig(configFile=None):
    info = {}
    if configFile is None:
        configFile = sys.argv[1] if (len(sys.argv) > 1) else input("Please enter the path of your config file: ")
    while not os.path.exists(configFile) or not configFile.endswith(".json"):
        configFile = input("Config JSON file not found! Please enter a valid path: ")
    with open(configFile) as f:
        info = json.load(f)
    return info

def statobj(data):
    obj = {}
    obj["avg"] = float(np.nanmean(data))
    obj["std"] = float(np.nanstd(data))
    obj["var"] = float(np.nanvar(data))
    obj["min"] = float(np.nanmin(data))
    obj["max"] = float(np.nanmax(data))
    obj["med"] = float(np.nanmedian(data))
    return obj

def sign(num):
    if num > 0:
        return '+'
    else:
        return '-'

def makeDir(key):
    if not os.path.exists(key):
        os.makedirs(key)

def makeTestDir(name):
    test_time = str(time.strftime("%y-%m-%d-%H%M"))
    makeDir("./test")
    dir_name = "./test/{}-{}".format(test_time, name)
    makeDir(dir_name)
    return dir_name

def string2arrPoint(key):
    positions = key.split()
    return [float(positions[i]) for i in range(len(positions))]

def arr2stringPoint(arr):
    s = ""
    for i in range(len(arr)):
        s = s + "{}{:.9f} ".format(sign(arr[i]), abs(arr[i]))
    return s[:-1]

# Adapted from https://stackoverflow.com/a/59591409
def read_pts(filename):
    """Read a .PTS landmarks file into a numpy ndarray"""
    with open(filename, 'rb') as f:
        # process the PTS header for n_rows and version information
        rows = version = None
        for line in f:
            if line.startswith(b"//"):  # comment line, skip
                continue
            header, _, value = line.strip().partition(b':')
            if not value:
                if header != b'{':
                    raise ValueError("Not a valid pts file")
                if version != 1:
                    raise ValueError(f"Not a supported PTS version: {version}")
                break
            try:
                if header == b"n_points":
                    rows = int(value)
                elif header == b"version":
                    version = float(value)  # version: 1 or version: 1.0
                elif not header.startswith(b"image_size_"):
                    # returning the image_size_* data is left as an excercise
                    # for the reader.
                    raise ValueError
            except ValueError:
                raise ValueError("Not a valid pts file")

        # if there was no n_points line, make sure the closing } line
        # is not going to trip up the numpy reader by marking it as a comment
        points = np.loadtxt(f, max_rows=rows, comments="}")

    if rows is not None and len(points) < rows:
        raise ValueError(f"Failed to load all {rows} points")
    return points

def process_bbox(points):
    points = points.T
    x, y = points[0], points[1]
    left, right = min(x), max(x)
    top, bottom = min(y), max(y)
    return left, top, right, bottom

def find_nc_PCA(dataset, n_c = 0, file = ""):
    X = dataset.data_df.iloc[:,:].values
    evr = np.cumsum(PCA(None).fit(X).explained_variance_ratio_)
    
    if n_c == 0:
        while evr[n_c] < 0.95: n_c += 1

    plt.plot(evr)
    plt.xlabel("Number of dimensions")
    plt.ylabel("% of total variance")
    plt.title("PCA Analysis for {} ({} at {})".format(
        dataset.name, np.around(evr[n_c], decimals=3), n_c
    ))
    
    if file != "":
        plt.savefig(file, dpi=600)
    else:
        plt.show(block=False)

    plt.clf()
    plt.close()

    X_pca = PCA(n_components=n_c).fit_transform(X)
    return n_c, X_pca