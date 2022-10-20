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
    makeDir("test")
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

def graph(xlabel, ylabel, data, data_dim = 1, line_count = 1, legend = [], file = "", marker=',', linestyle='-', title="", unit_size=2, width=6.4, height=4.8, hist=False, point_annotations=None, av_circle = False):

    figsize = (20,20) if av_circle else (width*unit_size,height*unit_size)
    plt.figure(figsize=figsize)
    fig, ax= plt.subplots(dpi=600)

    # add formatted labels
    titleFont = fm.FontProperties(fname="./static/fonts/KievitOffc-Bold.ttf",size='x-large')
    axisFont = fm.FontProperties(fname="./static/fonts/KievitOffc.ttf",size='x-large')
    legendFont = fm.FontProperties(fname="./static/fonts/KievitOffc-Ita.ttf",size='medium')

    if av_circle:
        plt.xlim(-2,2)
        plt.ylim(-1.5,1.5)  
        # draw the unit circle
        fig = plt.gcf()
        ax = fig.gca()
        circle1 = plt.Circle((0, 0), 1.0, color='0.25', fill=False)
        ax.add_artist(circle1)

        # print emotion labels
        emotionFont = fm.FontProperties(fname="./static/fonts/KievitOffc-BoldIta.ttf", size='large')
        ax.text(0.98, 0.35, 'Happy', fontproperties=emotionFont, size=10)
        ax.text(0.2, 1.05, 'Excited', fontproperties=emotionFont, size=10)
        ax.text(-1.3, 0.35, 'Afraid', fontproperties=emotionFont, size=10)
        ax.text(-0.7, 1, 'Angry', fontproperties=emotionFont, size=10)
        ax.text(-1.13, -0.55, 'Sad', fontproperties=emotionFont, size=10)
        ax.text(-0.9, -1, 'Depressed', fontproperties=emotionFont, size=10)
        ax.text(1, -0.25, 'Content', fontproperties=emotionFont, size=10)
        ax.text(0.7, -0.9, 'Calm', fontproperties=emotionFont, size=10) 

        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_position(("data",0))
        ax.spines["right"].set_position(("data",0))

    ax.set_xlabel(xlabel, fontproperties=axisFont)
    ax.set_ylabel(ylabel, fontproperties=axisFont)
    ax.set_title(title, fontproperties=titleFont)

    if not hist:
        if (data_dim == 1):
            for i in range(line_count):
                ax.plot(data[i], marker=marker, linestyle=linestyle)
        
        elif (data_dim == 2):
            if (line_count == 1):
                ax.plot(data[0], data[1], marker=marker, color="#D73F09", linestyle=linestyle)
            else:
                for i in range(line_count):
                    ax.plot(data[i][0], data[i][1], marker=marker, linestyle=linestyle)
    else:
        n, bins, patches = ax.hist(data, bins=int(1+3.3*np.log10(len(data))), facecolor="#D73F09")
    
    if point_annotations != None:
        for i in range(len(point_annotations)):
            if (i != 0 and i != len(point_annotations) - 1):
                ha = 'left' if (i % 2 == 0) else 'right'
                yoffset = -4 if (i % 2 == 0) else 4
                plt.annotate(point_annotations[i], # this is the text
                    (data[0][i],data[1][i]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(4,yoffset), # distance from text to points (x,y)
                    size='small',
                    fontweight='bold',
                    color='blue',
                    ha=ha) # horizontal alignment can be left, right or center

    if (legend != []):
        ax.legend(legend, prop=legendFont, fontsize='small')

    if (title != ""):
        plt.title(title)

    if (file != ""):
        plt.savefig(file, dpi=600)
    else:
        plt.show(block=False)
    
    plt.clf()
    plt.close()

def plot_AV_box(plots, labels, title="test", file="./test.png", plt_size=10, vert=True, showfliers=True):
    plt.figure(figsize=(plt_size, plt_size))
    fig, ax= plt.subplots(dpi=600)
    
    titleFont = fm.FontProperties(fname="./static/fonts/Stratum2-Bold.otf")
    ax.set_title(title, fontproperties=titleFont)    
    
    plt.boxplot(plots, labels=labels, showmeans=True, meanline=True, vert=vert, showfliers=showfliers)
    plt.tight_layout()
    
    plt.savefig(file, dpi=600)
    plt.show(block=False)
    plt.clf() 

def plot_AV_data(v, a, title="", colors="#D73F09", file="./test.png", plt_size=10, alpha=.5):
    plt.figure(figsize=(plt_size,plt_size))
    plt.scatter(v, a, s=20, c=colors, alpha=alpha)
    plt.xlim(-1.25,1.25)
    plt.ylim(-1.25,1.25)  

    # draw the unit circle
    fig = plt.gcf()
    ax = fig.gca()
    circle1 = plt.Circle((0, 0), 1.0, color='0.25', fill=False)
    ax.add_artist(circle1)

    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_position(("data",0))
    ax.spines["right"].set_position(("data",0))

    # add formatted labels
    titleFont = fm.FontProperties(fname="./static/fonts/Stratum2-Bold.otf")
    axisFont = fm.FontProperties(fname="./static/fonts/Stratum2-Medium.otf")
    emotionFont = fm.FontProperties(fname="./static/fonts/KievitOffc-BoldIta.ttf", size='xx-large')

    ax.set_xlabel("Valence", fontproperties=axisFont, size=plt_size*3)
    ax.set_ylabel("Arousal", fontproperties=axisFont, size=plt_size*3)
    ax.set_title(title, fontproperties=titleFont, size=plt_size*4)
    # ax.axes.xaxis.set_ticks([])
    # ax.axes.yaxis.set_ticks([])
    
    # print emotion labels
    ax.text(0.98, 0.35, 'Happy', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(0.5, 0.9, 'Excited', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(-1.16, 0.35, 'Afraid', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(-0.7, 0.9, 'Angry', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(-1.05, -0.25, 'Sad', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(-0.9, -0.9, 'Depressed', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(0.98, -0.25, 'Content', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(0.7, -0.9, 'Calm', fontproperties=emotionFont, size=int(plt_size*2.5)) 

    plt.savefig(file, dpi=600)
    plt.show(block=False)
    plt.clf() 
