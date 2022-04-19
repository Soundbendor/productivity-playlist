import hdf5_getters
import pandas as pd
import json
import pprint
import sys

deezer_path = "./data/deezer/deezer-spotify.csv"
msd_path = "../msd/data"
attr_track = "MSD_track_id"
attr_song = "MSD_sng_id"

json_data_out = "deezer_msd_data.json"
json_error_out = "deezer_msd_err.json"

songdata = pd.read_csv(deezer_path, header=0, index_col=0).dropna()
count = int(sys.argv[1]) if len(sys.argv) > 1 else len(songdata)

attributes = [k[4:] for k in filter(lambda x : x[:4] == 'get_', hdf5_getters.__dict__.keys())]
attributes.remove("num_songs")

msdata = {}
for a in attributes: msdata[a] = ["" for _ in range(count)]

errorlog = []

for i in range(count):
    # print(i, end='\r')
    song = songdata.iloc[i]
    trackid = song[attr_track]
    
    h5path = "{}/{}/{}/{}/{}.h5".format(msd_path, trackid[2], trackid[3], trackid[4], trackid)
    h5 = hdf5_getters.open_h5_file_read(h5path)

    for a in attributes:
        try:
            res = hdf5_getters.__getattribute__("get_" + a)(h5, 0)
            msdata[a][i] = res
        except:
            msdata[a][i] = None
            errorlog.append((i, a, trackid))

    h5.close()

pprint.pprint(msdata)
'''
json_data = json.dumps(msdata, indent=4)
with open(json_data_out, "w") as outfile:
    outfile.write(json_data)

json_error = json.dumps(errorlog, indent=4)
with open(json_error_out, "w") as outfile:
    outfile.write(json_error)
'''
