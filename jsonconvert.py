import pandas as pd
import pprint as pprint
import json

songdata = pd.read_csv("deam-data/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv", header=0, usecols=[0, 1, 3])
songobj = {}

def sign(num):
    if num > 0:
        return '+'
    else:
        return '-'

for i in range(len(songdata)):
    song = songdata.loc[i]
    pprint.pprint(song)
    print(song[0], song[1], song[2])

    songstring = "{}{} {}{}".format(sign(song[1]), abs(song[1]), sign(song[2]), abs(song[2]))

    if songstring not in songobj.keys():
        songobj[songstring] = []
    
    songobj[songstring].append(song[0])

pprint.pprint(songobj)

json_obj = json.dumps(songobj, indent=2)
with open("deampoints.json", "w") as outfile:
    outfile.write(json_obj)