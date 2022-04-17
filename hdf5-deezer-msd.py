import hdf5_getters
import pandas as pd

deezer_path = "./data/deezer/deezer-spotify.csv"
msd_path = "../msd/data"
attr_track = "MSD_track_id"
attr_song = "MSD_sng_id"

songdata = pd.read_csv(deezer_path, header=0, index_col=0).dropna()
non_one = 0

for i in range(len(songdata)):
    print(i, end='\r')
    song = songdata.iloc[i]
    trackid = song[attr_track]
    
    h5path = "{}/{}/{}/{}/{}.h5".format(msd_path, trackid[2], trackid[3], trackid[4], trackid)
    h5 = hdf5_getters.open_h5_file_read(h5path)

    count = hdf5_getters.get_num_songs(h5)
    if count > 1:
        non_one += 1

    h5.close()

print("Non one:", non_one)
