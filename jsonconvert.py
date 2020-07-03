import pandas as pd
import pprint as pprint
import json

songdata = pd.read_csv("deezer-spotify.csv", header=0, usecols=[0, 3, 4, 7])
songobj = {}

def sign(num):
    if num > 0:
        return '+'
    else:
        return '-'

for i in range(len(songdata)):
    song = songdata.loc[i]
    print(song.dzr_sng_id, song.valence, song.arousal)

    songstring = "{}{}{}{}".format(sign(song.valence), abs(song.valence), sign(song.arousal), abs(song.arousal))

    if songstring not in songobj.keys():
        songobj[songstring] = []
    
    songobj[songstring].append(song.dzr_sng_id)

pprint.pprint(songobj)

json_obj = json.dumps(songobj, indent=2)
with open("songpoints.json", "w") as outfile:
    outfile.write(json_obj)