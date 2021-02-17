import numpy as np
import json
import helper
import pandas as pd
from flask import Flask, render_template, url_for, request, redirect
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import prodplay
import sys
import time

sys.setdefaultencoding('utf-8')
app = Flask(__name__)

songdata = pd.read_csv("deezer-spotify.csv", header=0, index_col=0, usecols=[0,3,4,5,6,7], keep_default_na=False)
has_sp_id = [songdata.iloc[i][0] != "" for i in range(len(songdata))]
songdata = songdata[has_sp_id]

songpoints = {}
with open("deezer-points.json") as f:
    songpoints = json.load(f)

# coords = []
# for key in songpoints.keys():
#     coords.append(helper.string2arrPoint(key))
# coords = np.array(coords)

info = helper.loadConfig("./config.json")
sp = helper.Spotify(
    info["auth"]["client_id"], 
    info["auth"]["client_secret"], 
    info["auth"]["redirect_uri"], 
    info["auth"]["username"], 
    info["main"]["scope"]
)

coords = []
for i in range(len(songdata)):
    coords.append(songdata.iloc[i][3:].tolist())
coords = np.array(coords)

model = NearestNeighbors()
model.fit(coords)

song_arr = [ {
    "title": songdata.iloc[i][1],
    "artist": songdata.iloc[i][2],
    "arousal": np.around(songdata.iloc[i][3], decimals=2),
    "valence": np.around(songdata.iloc[i][4], decimals=2),
    "id": list(songdata.index.values)[i] 
} for i in range(32)]

@app.route('/')
def hello_world():
    return redirect(url_for('playlist'))

@app.route('/playlist', methods=['GET', 'POST'])
def playlist():
    list_arr = list_graph = song_orig = song_dest = n_songs = sp_link = None

    if len(request.args) == 3:
        song_orig = int(request.args['song_orig'])
        song_dest = int(request.args['song_dest'])
        n_songs = int(request.args['n_songs'])

    if song_orig != None and song_dest != None:
        songs, smooth, points = prodplay.makePlaylist(
            songdata, coords, song_orig, song_dest, n_songs, model, si = 3
        )

        list_arr = [{
            "title": songdata.loc[i][1],
            "artist": songdata.loc[i][2],
            "arousal": np.around(songdata.loc[i][3], decimals=2),
            "valence": np.around(songdata.loc[i][4], decimals=2)
        } for i in songs]

        # points = np.transpose(points)

        # test_time = str(time.strftime("%y-%m-%d_%H%M"))
        # helper.graph('valence', 'arousal', points, data_dim = 2, marker='.',
        #     file = 'static/playlist_{}.png'.format(test_time),
        #     title = "Path ({} songs) from ({}, {}) to ({}, {})".format(
        #         len(songs),
        #         np.around(songdata.loc[song_orig][3], decimals=2), np.around(songdata.loc[song_orig][4], decimals=2), 
        #         np.around(songdata.loc[song_dest][3], decimals=2), np.around(songdata.loc[song_dest][4], decimals=2), 
        #     )
        # )

        # list_graph = url_for('static', filename='playlist_{}.png'.format(test_time))

        track_ids = []
        for i in range(len(songs)):
            track_ids.append(songdata.loc[songs[i]][0])
        print(track_ids)
        title = "Flask Playlist"
        sp_link = "https://open.spotify.com/playlist/{}".format(helper.makeSpotifyList(sp, info["auth"]["username"], title, track_ids, False))

    orig = dest = None
    if song_orig != None:
        i = 0
        while i < len(song_arr):
            if song_arr[i]['id'] == song_orig: break
            else: i += 1
        if i < len(song_arr):
            orig = song_arr[i]
    
    if song_dest != None:
        i = 0
        while i < len(song_arr):
            if song_arr[i]['id'] == song_dest: break
            else: i += 1
        if i < len(song_arr):
            dest = song_arr[i]    

    if n_songs is None:
        n_songs = 10

    return render_template(
        'index.html', 
        graph=url_for('static', filename='test1.png'), 
        song_arr=song_arr,
        list_arr=list_arr,
        list_graph=list_graph,
        orig = orig,
        dest = dest,
        n = n_songs,
        sp_link = sp_link
    )
