import json
import pandas as pd
import os

# Designed to compare Arousal-Valence music datasets.
# Combines the following functionalities:
#   - find intersections (based on title & artist)

config = [
    {
        "name": "PmEmo",
        "csv": "data/Emo-Soundscapes/songs_and_annotations_2.csv",
        "cols": ["Artist","Song title", "mean_valence", "mean_arousal"],
    },
    {
        "name": "AMG",
        "csv": "data/AMG1608_release/with_spotify.csv",
        "cols": ["Artist","Song_title","Valence","Arousal"],
    },
    {
        "name": "Deezer",
        "csv": "data/deezer/deezer-spotify.csv",
        "cols": ["artist_name","track_name","valence","arousal"],
    },
    {
        "name": "DEAM",
        "csv": "data/deam/deam-spotify.csv",
        "cols": ["artist","title","valence_mean", "arousal_mean"],
    }
]

dirname = "intersect"
if not os.path.exists(dirname):
    os.makedirs(dirname)

namesets = []
for obj in config:
    df = pd.read_csv(obj["csv"], header=0, index_col=None, usecols=obj["cols"])[obj["cols"]]
    tuples = [(df.iloc[j, 0], df.iloc[j, 1]) for j in range(len(df))]
    namesets.append(set(tuples))

for i in range(len(namesets) - 1):
    for j in range(i + 1, len(namesets)):
        intersection = namesets[i] & namesets[j]
        print(f"Intersection between {config[i]['name']} and {config[j]['name']} = {len(intersection)} songs")

        if len(intersection) > 0:
            df_int = pd.DataFrame(list(intersection), columns=["artist", "title"])
            df_int.to_csv(f"{dirname}/int_{config[i]['name']}_{config[j]['name']}.csv")