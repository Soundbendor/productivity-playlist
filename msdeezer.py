import hdf5_getters
import pandas as pd
import json
import pprint
import sys

deezer_path = "./data/deezer/deezer-spotify.csv"
msd_path = "../msd/data"
attr_track = "MSD_track_id"
attr_song = "MSD_sng_id"

songdata = pd.read_csv(deezer_path, header=0, index_col=0).dropna()
count = int(sys.argv[1]) if len(sys.argv) > 1 else len(songdata)

# attributes = [
#  "artist_7digitalid",
#  "artist_familiarity",
#  "artist_hotttnesss",
#  "artist_mbid",
#  "artist_playmeid",
#  "audio_md5",
#  "danceability",
#  "energy",
#  "key",
#  "key_confidence",
#  "loudness",
#  "mode",
#  "mode_confidence",
#  "tempo",
#  "time_signature",
#  "time_signature_confidence",
#  "track_7digitalid"
# ]
attributes = [k[4:] for k in filter(lambda x : x[:4] == 'get_', hdf5_getters.__dict__.keys())]

# msdata = {}
# for a in attributes: msdata[a] = ["" for _ in range(count)]

for i in range(count):
    song = songdata.iloc[i]
    trackid = song[attr_track]
    
    h5path = "{}/{}/{}/{}/{}.h5".format(msd_path, trackid[2], trackid[3], trackid[4], trackid)
    h5 = hdf5_getters.open_h5_file_read(h5path)

    print("Grabbed file {}".format(h5path))

    for a in attributes:
        print("\n{}:".format(a))
        try:
            res = hdf5_getters.__getattribute__("get_" + a)(h5, 0)
            pprint.pprint(res)
            # msdata[a][i] = res
        except:
            # msdata[a][i] = None
            print("{0:>6}: Error finding {}".format(i, a))

    h5.close()

# for a in attributes:
#     songdata[a] = msdata[a]
# songdata.to_csv(path_or_buf="msdeezer.csv")

