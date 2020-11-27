import json
import pandas as pd
import numpy as np
import pprint

match_scores = {}
with open("data/LMD/match_scores.json") as f:
    match_scores = json.load(f)

songdata = pd.read_csv(
    "data/deezer-data/deezer-spotify.csv",
    header=0, usecols=[0,2]
)

DZR_keys = set(songdata["MSD_track_id"].tolist())
LMD_keys = set([k for k in match_scores.keys()])
INT_keys = DZR_keys & LMD_keys

json_obj = json.dumps(list(INT_keys), indent=2)
with open("data/LMD/deezer_intersection.json", "w") as f:
    f.write(json_obj)