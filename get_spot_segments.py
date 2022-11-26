import spotify
import helper
from songdataset import SongDataset
from pprint import pprint
import json
import pandas as pd
import numpy as np

info        = helper.loadConfig("config.json")
datasetpath = "data/deezer/deezer-std-all.csv"

sp, spo = spotify.Spotify(
    info["auth"]["client_id"], 
    info["auth"]["client_secret"], 
    info["auth"]["redirect_uri"], 
    info["auth"]["username"], 
    info["auth"]["scope"]
)

feats_ind = ["loudness_max", "loudness_start"]
feats_arr = [("pitches", 12), ("timbre", 12)]

def fillcols(s, outcols):
    for feat in feats_ind: outcols["{}_{}".format(s, feat)] = []
    for feat, n in feats_arr: 
        for i in range(n):
            formatstr = "{}_{}_{}".format(s, feat, i)
            outcols[formatstr] = []
    return outcols

'''
    We want to take the segments arrays and do a weighted average.
    - weight by order of segments inversely.
        - We can weigh this as [N, N-1, ..., 2, 1] / n(n+1)/2.
    - weight by duration of segments. 
        - We can weigh by duration of segments / total duration.
'''
def weighted_avg(arr, dur, sum, len):
    tot = 0.0
    den = (len * (len + 1)) // 2
    
    for i in range(len):
        ord = len - i
        tot += (arr[i] * dur[i] * ord)

    return (tot / (den * sum))

def grab_segment_data(segments, mode = "cnt", num = 10):
    outvals = {}

    ## Weighted averages!
    segs = {"head": segments[0:num], 
            "tail": segments[(-1*num):]}
    durs = {"head": [seg["duration"] for seg in segs["head"]], 
            "tail": [seg["duration"] for seg in segs["tail"]]}
    sums = {"head": sum(durs["head"]), 
            "tail": sum(durs["tail"])}
    lens = {"head": len(durs["head"]), 
            "tail": len(durs["tail"])}

    for s in ["head", "tail"]:
        for feat in feats_ind: 
            coln = "{}_{}".format(s, feat)
            data = [seg[feat] for seg in segs[s]]
            wavg = weighted_avg(data, durs[s], sums[s], lens[s])
            outvals[coln] = np.around(wavg, decimals=6)
            
        for feat, n in feats_arr: 
            for i in range(n):
                coln = "{}_{}_{}".format(s, feat, i)
                data = [seg[feat][i] for seg in segs[s]]
                wavg = weighted_avg(data, durs[s], sums[s], lens[s])
                outvals[coln] = np.around(wavg, decimals=6)

    return outvals

def grab_dataset(outpath):
    testlength  = 1
    songdata = SongDataset(
        name="Deezer",
        cols=info["cols"]["deezer"],
        path=datasetpath, knn=True, verbose=True,
        data_index = 5, arousal = 4, valence = 3,
    )

    df = songdata.full_df[0:testlength].copy()
    outcols = {}
    fillcols("head", outcols)
    fillcols("tail", outcols)

    for i in range(testlength): 
        spid = df.iloc[i]["sp_track_id"]
        analysis = sp.audio_analysis(spid)
        segments = analysis["segments"]
        print(i, end="\r")

        vals = grab_segment_data(spid, mode = "cnt", num = 10)
        for key in vals: outcols[key].append(vals[key])

    # EXPORT DATA TO CSV!!!
    for col in outcols: df[col] = outcols[col]
    df.to_csv(outpath)












