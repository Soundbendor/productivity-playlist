import spotify
import helper
from songdataset import SongDataset
from pprint import pprint
import json
import pandas as pd

info = helper.loadConfig("config.json")
datasetpath = "data/deezer/deezer-std-all.csv"

sp, spo = spotify.Spotify(
    info["auth"]["client_id"], 
    info["auth"]["client_secret"], 
    info["auth"]["redirect_uri"], 
    info["auth"]["username"], 
    info["auth"]["scope"]
)

songdata = SongDataset(
    name="Deezer+Spotify",
    cols=info["cols"]["deezer"] + info["cols"]["spotify"],
    path=datasetpath, knn=True, verbose=True,
    data_index = 5, arousal = 4, valence = 3,
)

testlen = 100
spids = songdata.full_df["sp_track_id"][0:testlen].tolist()
print(len(spids))
slens = []

pstds = []
pmins = []
pmaxs = []
pavgs = []
pmeds = []

tstds = []
tmins = []
tmaxs = []
tavgs = []
tmeds = []


for i in range(testlen): 
    spid = spids[i]
    analysis = sp.audio_analysis(spid)
    segments = analysis["segments"]

    print(i, end="\r")
    
    # For each song, we want:
    ## number of segments
    slens.append(len(segments))
    # print("Num segments: ", len(segments))

    ## spread of lengths of pitches and timbres (should be 0)
    lps = []
    lts = []
    
    for s in segments:
        lps.append(len(s["pitches"]))
        lts.append(len(s["timbre"]))

    pstats = helper.statobj(lps)
    pstds.append(pstats["std"])
    pmins.append(pstats["min"])
    pmaxs.append(pstats["max"])
    pavgs.append(pstats["avg"])
    pmeds.append(pstats["med"])

    # print("Pitches stats: ", pstats)

    tstats = helper.statobj(lts)
    tstds.append(tstats["std"])
    tmins.append(tstats["min"])
    tmaxs.append(tstats["max"])
    tavgs.append(tstats["avg"])
    tmeds.append(tstats["med"])

    # print("Timbres stats: ", pstats)

outobj = {}
outobj["slens"] = helper.statobj(slens)

outobj["pstds"] = helper.statobj(pstds)
outobj["pmins"] = helper.statobj(pmins)
outobj["pmaxs"] = helper.statobj(pmaxs)
outobj["pavgs"] = helper.statobj(pavgs)
outobj["pmeds"] = helper.statobj(pmeds)

outobj["tstds"] = helper.statobj(tstds)
outobj["tmins"] = helper.statobj(tmins)
outobj["tmaxs"] = helper.statobj(tmaxs)
outobj["tavgs"] = helper.statobj(tavgs)
outobj["tmeds"] = helper.statobj(tmeds)

outdf = songdata.full_df["sp_track_id"][0:testlen].copy()

#### FIGURE OUT WHY THIS DOESN'T WORK !!!
outdf["slens"] = pd.Series(slens)

outdf["pstds"] = pd.Series(pstds)
outdf["pmins"] = pd.Series(pmins)
outdf["pmaxs"] = pd.Series(pmaxs)
outdf["pavgs"] = pd.Series(pavgs)
outdf["pmeds"] = pd.Series(pmeds)

outdf["tstds"] = pd.Series(tstds)
outdf["tmins"] = pd.Series(tmins)
outdf["tmaxs"] = pd.Series(tmaxs)
outdf["tavgs"] = pd.Series(tavgs)
outdf["tmeds"] = pd.Series(tmeds)

helper.jsonout(outobj, "out/seg-len-stats.json")
outdf.to_csv("out/deezer-spot-segment-stats.csv")