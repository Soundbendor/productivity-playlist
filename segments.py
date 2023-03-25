import spotify
import helper
from pprint import pprint
import plot
import pandas as pd
import numpy as np

FEATS_IND = ["loudness_max", "loudness_start"]
FEATS_ARR = [("pitches", 12), ("timbre", 12)]

def fillcols(s, outcols, size=0):
    outcols[f"{s}_duration"] = [None] * size
    outcols[f"{s}_num_samples"] = [None] * size
    for feat in FEATS_IND: outcols["{}_{}".format(s, feat)] = [None] * size
    for feat, n in FEATS_ARR: 
        for i in range(n):
            formatstr = "{}_{}_{}".format(s, feat, i)
            outcols[formatstr] = [None] * size
    return outcols

'''
    We want to take the segments arrays and do a weighted average.
    - weight by order of segments inversely.
        - We can weigh this as [N, N-1, ..., 2, 1] / n(n+1)/2.
    - weight by duration of segments. 
        - We can weigh by duration of segments / total duration.
'''
def weighted_avg(arr, dur, time, n):
    total = 0.0
    denom = 0.0
    
    for i in range(n):
        index = n - i
        coeff = dur[i] * index
        denom += coeff
        total += (arr[i] * coeff)

    return (total / denom)

def grab_segment_data(segments, mode = "cnt", num = 10.0):
    outvals = {}
    segvals = {}

    segs = {"head": [], 
            "tail": []}
    if mode == "cnt":
        segs["head"] = segments[0:num] 
        segs["tail"] = segments[-1:(-1*num-1):-1]
    
    elif mode == "dur":
        head_dur, tail_dur = 0.0, 0.0
        head_idx, tail_idx = 0, len(segments) - 1
        
        while head_dur < num and head_idx < len(segments):
            segs["head"].append(segments[head_idx])
            head_dur += segments[head_idx]["duration"]
            head_idx += 1

        while tail_dur < num and tail_idx >= 0:
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

    for s in ["head", "tail"]:
        # print("---- {}: len = {}, dur = {} ----".format(s, lens[s], sums[s]))
        # pprint(segs[s])
        segvals["{}_duration".format(s)] = durs[s]
        
        for feat in FEATS_IND: 
            coln = "{}_{}".format(s, feat)
            data = [seg[feat] for seg in segs[s]]
            wavg = weighted_avg(data, durs[s], sums[s], lens[s])
            outvals[coln] = np.around(wavg, decimals=6)
            segvals[coln] = data
            
        for feat, n in FEATS_ARR: 
            for i in range(n):
                coln = "{}_{}_{}".format(s, feat, i)
                data = [seg[feat][i] for seg in segs[s]]
                wavg = weighted_avg(data, durs[s], sums[s], lens[s])
                outvals[coln] = np.around(wavg, decimals=6)
                segvals[coln] = data

        outvals[f"{s}_duration"] = np.around(sums[s], 5)
        outvals[f"{s}_num_samples"] = lens[s]

    # print("---- Output values ----")
    # pprint(outvals)
    return segvals, outvals

def grab_dataset(outpath, length, mode, num, df_in):
    df = df_in[0:length].copy()
    outcols = {}
    fillcols("head", outcols)
    fillcols("tail", outcols)

    for i in range(length): 
        spid = df.iloc[i]["sp_track_id"]
        
        try:
            analysis = sp.audio_analysis(spid)
            segments = analysis["segments"]
            print("{} / {}".format(i, length), end="\r")

            _, vals = grab_segment_data(segments, mode, num)
            for key in vals: outcols[key].append(vals[key])
        
        except:
            for key in outcols: outcols[key].append(None) 

    # EXPORT DATA TO CSV!!!
    for col in outcols: df[col] = outcols[col]
    df.to_csv(outpath)

def grab_datasets(df_in, orders, retry = True):
    all_outcols = []
    for _ in range(len(orders)):
        outcols = {}
        fillcols("head", outcols, len(df_in))
        fillcols("tail", outcols, len(df_in))
        all_outcols.append(outcols)

    def fout():
        for i in range(len(orders)):
            mode, num = orders[i]
            outcols = all_outcols[i]
            df_out = df_in.copy()
            for col in outcols: df_out[col] = outcols[col]
            df_out.to_csv("./data/deezer/segments/{}{:03}.csv".format(mode, num))  

    def fill(analysis):
        segments = analysis["segments"]
        for j, (mode, num) in enumerate(orders):
            realnum = analysis["track"][("duration" if mode == 'dur' else "num_samples")]
            _, vals = grab_segment_data(segments, mode, min(num, realnum))
            for key in vals: all_outcols[j][key][i] = vals[key]
    
    try_idxs = list(range(len(df_in)))
    if retry:
        try_idxs = []
        for j, (mode, num) in enumerate(orders):
            path = "./data/deezer/segments/{}{:03}.csv".format(mode, num)
            df = pd.read_csv(path)
            for c in all_outcols[j]: all_outcols[j][c] = df[c].tolist()
            nulls = pd.isnull(df).any(axis=1).tolist()
            for i in range(len(nulls)):
                if nulls[i]: try_idxs.append(i)
        try_idxs = list(set(try_idxs))

    success = True
    while len(try_idxs) > 0 and success:
        l = len(try_idxs)
        fail_idxs = []
        # spotify.refresh_token(spo)

        for k, i in enumerate(try_idxs): 
            spid = df_in.iloc[i]["sp_track_id"]
            try:
                print("{} / {} ... failed on {:05}".format(k, len(try_idxs), len(fail_idxs)), end="\r")
                analysis = sp.audio_analysis(spid)
            except:
                spotify.refresh_token(spo)
                fail_idxs.append(i)
            else:
                fill(analysis)

            if len(try_idxs) >= 100 and k % min(500, len(try_idxs) // 5) == 0:
                fout()   

        fout()    
        print(f"\nFailed on {len(fail_idxs)} indices")
        if len(fail_idxs) == l: success = False
        try_idxs = fail_idxs        

    fout()
    return fail_idxs 

def fill_segment_datasets(dfs, path, mode, num):    
    for index in df.index.values.tolist():
        spid = df.loc[index]["sp_track_id"]
        hlmx = df.loc[index]["head_loudness_max"]

        if pd.isnull(hlmx):
            print(f"Looking for {spid} ... ", end="")
            try:
                analysis = sp.audio_analysis(spid)
                segments = analysis["segments"]
                _, vals = grab_segment_data(segments, mode, num)
                for key in vals: df.at[index, key] = vals[key]
                print("Success!")
            except:
                print("Failed :(")
    
    df.to_csv(path)
    return df

def test_segcounts(spid, mode, num):

    test_range = [x+1 for x in range(num)]
    segments = sp.audio_analysis(spid)["segments"]
    dirname = helper.makeTestDir("seg{}{:03}".format(mode, num))

    testcols = {}
    fillcols("head", testcols)
    fillcols("tail", testcols)  

    for r in test_range:
        # print("\n\n---- {}: {} ----\n".format(mode, r))
        _, vals = grab_segment_data(segments, mode, num = r)
        for key in vals: testcols[key].append(vals[key])

    testdf = pd.DataFrame(testcols)
    testdf.to_csv("{}/test.csv".format(dirname))

    if mode == "cnt":
        datacols, _ = grab_segment_data(segments, mode, num)
        datadf = pd.DataFrame(datacols)
        datadf.to_csv("{}/data.csv".format(dirname))
    
    for col in testcols:
        data = []
        if mode == "cnt":
            data = [[test_range, testcols[col]], [test_range, datacols[col]]]
        else:
            data = [test_range, testcols[col]]
        count = 2 if mode == "cnt" else 1
        
        plot.line("# of segments/seconds", col, data, 
                    count = count, dim = 2,
                    title = "Averages of {}".format(col),
                    file="{}/{}.png".format(dirname, col))

if __name__ == "__main__":     
    info        = helper.loadConfig("config.json")
    datasetpath = "data/deezer/deezer-std-all.csv"
    generate    = [
        ("dur", 60), ("cnt", 200),
        ("dur", 50), ("cnt", 150),
        ("dur", 40), ("cnt", 100),
        ("dur", 30), ("cnt", 75),
        ("dur", 20), ("cnt", 50),
        ("dur", 10), ("cnt", 25),
        ("dur", 5), ("cnt", 10),
        ("dur", 2), ("cnt", 5),
        ("dur", 1), ("cnt", 1),
    ]

    sp, spo = spotify.Spotify(
        info["auth"]["client_id"], 
        info["auth"]["client_secret"], 
        info["auth"]["redirect_uri"], 
        info["auth"]["username"], 
        info["auth"]["scope"]
    )

    songdata = pd.read_csv(datasetpath, usecols=info["cols"]["deezer"])
    fails = grab_datasets(songdata, generate)
