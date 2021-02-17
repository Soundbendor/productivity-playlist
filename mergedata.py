import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
import helper

info = helper.loadConfig()
deezer_info = info["mergeData"]["deezer"]
deam_info = info["mergeData"]["deam"]

deezer = pd.read_csv(deezer_info["csv"], header=0, index_col=0, usecols=deezer_info["cols"])
deam = pd.read_csv(deam_info["csv"], header=0, index_col=0, usecols=deam_info["cols"])

inter = []
with open(info["mergeData"]["int"]) as f: 
    inter = json.load(f)

print(deezer.head())
print(deam.head())

deam_ids = []
deezer_ids = []
for musician, song in inter:
    deezer_obj = deezer.query("artist_name == \"{}\" and track_name == \"{}\"".format(musician, song)).iloc[0]
    deam_obj = deam.query("artist == \"{}\" and title == \"{}\"".format(musician, song)).iloc[0]
    deezer_idx = deezer.query("artist_name == \"{}\" and track_name == \"{}\"".format(musician, song)).index[0]
    deam_idx = deam.query("artist == \"{}\" and title == \"{}\"".format(musician, song)).index[0]

    print(deezer_obj)
    print(deam_obj)
    print("\n\n\n")

    deezer_ids.append(deezer_idx)
    deam_ids.append(deam_idx)

print(deezer_ids, deam_ids)

deezer_copy = deezer.copy(deep=True)
scaler = MinMaxScaler(feature_range=(-1,1))
scaled_values = scaler.fit_transform(deezer.iloc[:, 0:2])

for i in range(len(scaled_values)):
    deezer_copy.iat[i,0] = scaled_values[i][0]
    deezer_copy.iat[i,1] = scaled_values[i][1]

deam_copy = deam.copy(deep=True)
scaler = MinMaxScaler(feature_range=(-1,1))
scaled_values = scaler.fit_transform(deam.iloc[:, 0:2])

for i in range(len(scaled_values)):
    deam_copy.iat[i,0] = scaled_values[i][0]
    deam_copy.iat[i,1] = scaled_values[i][1]

for i in range(len(inter)):
    print("{} - {}:".format(inter[i][0], inter[i][1]))

    song = deezer.loc[deezer_ids[i]]
    print(" - Deezer: {} {}".format(song[0], song[1]))
    song = deam.loc[deam_ids[i]]
    print(" - DEAM: {} {}".format(song[0], song[1]))
    song = deezer_copy.loc[deezer_ids[i]]
    print(" - Deezer CP: {} {}".format(song[0], song[1]))
    song = deam_copy.loc[deam_ids[i]]
    print(" - DEAM CP: {} {}".format(song[0], song[1]))
    print()