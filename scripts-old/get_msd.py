import hdf5_getters
import pandas as pd
import numpy as np
import json
import pprint
import sys

deezer_path = "./msdeezer.csv"
msd_path = "../msd/data"
attr_track = "MSD_track_id"
attr_song = "MSD_sng_id"

songdata = pd.read_csv(deezer_path, header=0, index_col=0).dropna()
count = int(sys.argv[1]) if len(sys.argv) > 1 else len(songdata)
msdata = {}

# attributes = [k[4:] for k in filter(lambda x : x[:4] == 'get_', hdf5_getters.__dict__.keys())]
# attributes.remove("num_songs")

attributes = [
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
]
for a in attributes: msdata[a] = ["" for _ in range(count)]

array_attributes = [
    ("segments_loudness_max",1),
    ("segments_loudness_max_time",1),
    ("segments_timbre",12)
]
for a, n in array_attributes: 
    if n == 1:
        msdata[a + "_avg"] = ["" for _ in range(count)]
        msdata[a + "_std"] = ["" for _ in range(count)]
        msdata[a + "_var"] = ["" for _ in range(count)]
        msdata[a + "_min"] = ["" for _ in range(count)]
        msdata[a + "_max"] = ["" for _ in range(count)]
        msdata[a + "_med"] = ["" for _ in range(count)]
    
    else:
        for i in range(n):
            msdata[a + "_" + str(i) + "_avg"] = ["" for _ in range(count)]
            msdata[a + "_" + str(i) + "_std"] = ["" for _ in range(count)]
            msdata[a + "_" + str(i) + "_var"] = ["" for _ in range(count)]
            msdata[a + "_" + str(i) + "_min"] = ["" for _ in range(count)]
            msdata[a + "_" + str(i) + "_max"] = ["" for _ in range(count)]
            msdata[a + "_" + str(i) + "_med"] = ["" for _ in range(count)]            

for i in range(count):
    song = songdata.iloc[i]
    trackid = song[attr_track]
    
    h5path = "{}/{}/{}/{}/{}.h5".format(msd_path, trackid[2], trackid[3], trackid[4], trackid)
    h5 = hdf5_getters.open_h5_file_read(h5path)

    # print("Grabbed file {}".format(h5path))

    for a in attributes:
        # print("\n{}:".format(a))
        try:
            res = hdf5_getters.__getattribute__("get_" + a)(h5, 0)
            msdata[a][i] = res
        except:
            msdata[a][i] = None
            # print("{0:>6}: Error finding {}".format(i, a))

    for a, n in array_attributes:
        try:
            res = hdf5_getters.__getattribute__("get_" + a)(h5, 0)

            if n == 1:
                msdata[a + "_avg"][i] = float(np.nanmean(res))
                msdata[a + "_std"][i] = float(np.nanstd(res)) 
                msdata[a + "_var"][i] = float(np.nanvar(res)) 
                msdata[a + "_min"][i] = float(np.nanmin(res)) 
                msdata[a + "_max"][i] = float(np.nanmax(res)) 
                msdata[a + "_med"][i] = float(np.nanmedian(res)) 
            
            else:
                res = np.transpose(res)
                for j in range(n):
                    msdata[a + "_" + str(j) + "_avg"][i] = float(np.nanmean(res[j])) 
                    msdata[a + "_" + str(j) + "_std"][i] = float(np.nanstd(res[j])) 
                    msdata[a + "_" + str(j) + "_var"][i] = float(np.nanvar(res[j])) 
                    msdata[a + "_" + str(j) + "_min"][i] = float(np.nanmin(res[j])) 
                    msdata[a + "_" + str(j) + "_max"][i] = float(np.nanmax(res[j])) 
                    msdata[a + "_" + str(j) + "_med"][i] = float(np.nanmedian(res[j]))      

        except:
            msdata[a][i] = None
            # print("{0:>6}: Error finding {}".format(i, a))

    h5.close()

for a in msdata.keys():
    songdata["MSD_" + a] = msdata[a]
songdata.to_csv(path_or_buf="msdeezerplus.csv")

