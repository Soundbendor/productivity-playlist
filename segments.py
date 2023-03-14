import spotify
import helper
from pprint import pprint
import plot
import pandas as pd
import numpy as np

FEATS_IND = ["loudness_max", "loudness_start"]
FEATS_ARR = [("pitches", 12), ("timbre", 12)]

def fillcols(s, outcols):
    for feat in FEATS_IND: outcols["{}_{}".format(s, feat)] = []
    for feat, n in FEATS_ARR: 
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

def fill_segment_datasets(df, path, mode, num):    
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
    generate    = [("dur", 30), ("cnt", 100)]

    sp, spo = spotify.Spotify(
        info["auth"]["client_id"], 
        info["auth"]["client_secret"], 
        info["auth"]["redirect_uri"], 
        info["auth"]["username"], 
        info["auth"]["scope"]
    )

    songdata = pd.read_csv(datasetpath, cols=info["cols"]["deezer"])

    # randidx = np.random.randint(0, len(songdata))
    # randsong = songdata.full_df.iloc[randidx]
    # print(randsong)

    # for mode, num in generate:
    #     test_segcounts(randsong["sp_track_id"], mode, num)
    #     grab_dataset("out/{}-segments-{}{:03}.csv".format(songdata.name, mode, num), len(songdata), mode, num)

    DEEZER_SEG_100  = "./data/deezer/deezer-segments-cnt100.csv"
    DEEZER_SEG_D30  = "./data/deezer/deezer-segments-dur030.csv"    

    OLD_DEEZER_SEG_100  = "./data/deezer/deezer-segments-cnt100-old.csv"
    OLD_DEEZER_SEG_D30  = "./data/deezer/deezer-segments-dur030-old.csv"    

    fix = [
        ("dur", 30, OLD_DEEZER_SEG_D30, DEEZER_SEG_D30), 
        ("cnt", 100, OLD_DEEZER_SEG_100, DEEZER_SEG_100)
    ]

    for mode, num, old, new in fix:
        df = pd.read_csv(old, header=0, index_col=0)
        desc_stats = pd.DataFrame({
            'Missing Values': df.isnull().sum(),
            'Mean': df.mean(),
            'Median': df.median(),
            'Mode': df.mode().iloc[0]
        })
        print(desc_stats)
        fill_segment_datasets(df, new, mode, num)
        desc_stats = pd.DataFrame({
            'Missing Values': df.isnull().sum(),
            'Mean': df.mean(),
            'Median': df.median(),
            'Mode': df.mode().iloc[0]
        })
        print(desc_stats)











