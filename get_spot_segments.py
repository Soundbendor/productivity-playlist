import spotify
import helper
from songdataset import SongDataset
from pprint import pprint
import plot
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

songdata = SongDataset(
    name="Deezer",
    cols=info["cols"]["deezer"],
    path=datasetpath, knn=True, verbose=True,
    data_index = 5, arousal = 4, valence = 3,
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
def weighted_avg(arr, dur, time, n):
    tot = 0.0
    den = (n * (n + 1)) // 2
    
    for i in range(n):
        idx = n - i
        tot += (arr[i] * dur[i] * idx)

    return (tot / (den * time))
    # return (tot / (den * sum))

def grab_segment_data(segments, mode = "cnt", num = 10.0):
    outvals = {}

    segs = {"head": [], 
            "tail": []}
    if mode == "cnt":
        segs["head"] = segments[0:num] 
        segs["tail"] = segments[(-1*num):]
    
    elif mode == "dur":
        head_dur, tail_dur = 0.0, 0.0
        head_idx, tail_idx = 0, len(segments) - 1
        
        while head_dur < num:
            segs["head"].append(segments[head_idx])
            head_dur += segments[head_idx]["duration"]
            head_idx += 1

        while tail_dur < num:
            segs["tail"].append(segments[tail_idx])
            tail_dur += segments[tail_idx]["duration"]
            tail_idx -= 1
    
    # Get supplemental data for weighted averages.
    durs = {"head": [seg["duration"] for seg in segs["head"]], 
            "tail": [seg["duration"] for seg in segs["tail"]]}
    sums = {"head": sum(durs["head"]), 
            "tail": sum(durs["tail"])}
    lens = {"head": len(durs["head"]), 
            "tail": len(durs["tail"])}

    # print("sums:", sums, "\nlens:", lens)

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

def grab_dataset(outpath, length):
    df = songdata.full_df[0:length].copy()
    outcols = {}
    fillcols("head", outcols)
    fillcols("tail", outcols)

    for i in range(length): 
        spid = df.iloc[i]["sp_track_id"]
        
        try:
            analysis = sp.audio_analysis(spid)
            segments = analysis["segments"]
            print("{} / {}".format(i, length), end="\r")

            vals = grab_segment_data(segments, mode = "dur", num = 5.0)
            for key in vals: outcols[key].append(vals[key])
        
        except:
            for key in outcols: outcols[key].append(None) 

    # EXPORT DATA TO CSV!!!
    for col in outcols: df[col] = outcols[col]
    df.to_csv(outpath)

def test_segcounts(spid):
    test_range = [x+1 for x in range(30)]
    segments = sp.audio_analysis(spid)["segments"]
    dirname = helper.makeTestDir("segcounts")

    testcols = {}
    fillcols("head", testcols)
    fillcols("tail", testcols)  
      
    for r in test_range:
        vals = grab_segment_data(segments, mode = "dur", num = r)
        for key in vals: testcols[key].append(vals[key])
    
    for col in testcols:
        plot.line("# of segments/seconds", col, [test_range, testcols[col]], 
                    count = 1, dim = 2,
                    title = "Spread of {}".format(col),
                    file="{}/{}.png".format(dirname, col))
        
randidx = np.random.randint(0, len(songdata))
randsong = songdata.full_df.iloc[randidx]
print(randsong)
test_segcounts(randsong["sp_track_id"])

# grab_dataset("out/deezer-segments.csv", 100)











