import numpy as np
import json
import helper
import spotify
import pandas as pd
from flask import Flask, render_template, url_for, request, redirect
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import prodplay
import sys
import time
import pprint
from songdataset import SongDataset

app = Flask("prodplay")
path = "static/deezer-std-all.csv"
info = helper.loadConfig("./config.json")

dataset = SongDataset(
    name="Deezer+Spotify",
    cols=info["cols"]["deezer"] + info["cols"]["spotify"],
    path=path, knn=True, verbose=True,
    data_index = 5, arousal = 4, valence = 3,
)
dataset.make_knn()

sp, spo = spotify.Spotify(
    info["auth"]["client_id"], 
    info["auth"]["client_secret"], 
    info["auth"]["redirect_uri"], 
    info["auth"]["username"], 
    info["auth"]["scope"],
    auto=True
)

song_arr = [ {
    "title": dataset.full_df.iloc[i][1],
    "artist": dataset.full_df.iloc[i][2],
    "arousal": np.around(dataset.full_df.iloc[i][3], decimals=2),
    "valence": np.around(dataset.full_df.iloc[i][4], decimals=2),
    "id": list(dataset.full_df.index.values)[i] 
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
        songs, points, feats, smooths, steps = prodplay.makePlaylist(
            dataset, song_orig, song_dest, n_songs
        )
        print("Got the songs")

        list_arr = [{
            "title": dataset.full_df.loc[i][2],
            "artist": dataset.full_df.loc[i][1],
            "arousal": np.around(dataset.full_df.loc[i][4], decimals=2),
            "valence": np.around(dataset.full_df.loc[i][3], decimals=2)
        } for i in songs]
        print("Listed the songs")

        track_ids = [dataset.get_spid(i) for i in songs]
        title = "Playlist {}".format(str(time.strftime("%Y-%m-%d %H:%M")))
        sp_link = "https://open.spotify.com/playlist/{}".format(
            spotify.makePlaylist(sp, spo, title, track_ids, False))
        print("Created playlist")

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
        n_songs = 5

    return render_template(
        'index.html', 
        song_arr=song_arr,
        list_arr=list_arr,
        list_graph=list_graph,
        orig = orig,
        dest = dest,
        n = n_songs,
        sp_link = sp_link
    )