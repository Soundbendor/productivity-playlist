import streamlit as st
import altair as alt
import numpy as np
import random
import pandas as pd
import json
import pprint
import time
import sys
import os
import math
import warnings

#our modules
import helper
import prodplay
import spotify
import algos
from songdataset import SongDataset

st.title("Mood-Dynamic Playlist")
st.write("By [Shaurya Gaur](https://shaurgaur.github.io/)")
st.write("For [Soundbendor Lab](https://soundbendor.org/), Oregon State University")
st.write("Last updated November 2nd, 2022")

#get important personal information from Spotify API
datasetpath = "static/deezer-std-all.csv"
info = helper.loadConfig("config.json")
dataset = SongDataset(
    name="Deezer+Spotify",
    cols=info["cols"]["deezer"] + info["cols"]["spotify"],
    path=datasetpath, knn=True, verbose=True,
    feat_index = 5, arousal = 4, valence = 3,
)

sp, spo = spotify.Spotify(
    info["auth"]["client_id"], 
    info["auth"]["client_secret"], 
    info["auth"]["redirect_uri"], 
    info["auth"]["username"], 
    info["auth"]["scope"]
)

songoptionobj = {
    "label": [],
    "id": []
}

for i in range(64):
    songoptionobj["label"].append("({},{}), id: {} ... {} - {} ".format(
        np.around(dataset.full_df.iloc[i]["valence"], decimals=2),
        np.around(dataset.full_df.iloc[i]["arousal"], decimals=2),       
        list(dataset.full_df.index.values)[i],
        dataset.full_df.iloc[i]["artist_name"],
        dataset.full_df.iloc[i]["track_name"],
    ))
    songoptionobj["id"].append(list(dataset.full_df.index.values)[i])

optiondf = pd.DataFrame(songoptionobj)

st.write("## Let's make a playlist!")
form = st.form("playlist_form")
origstr = form.selectbox("Choose a song to start with:", optiondf, index=52)
deststr = form.selectbox("Choose a song to end with:", optiondf, index=8)
nsongs = form.number_input("Number of songs:", 3, 13, value=7, step=1)
spotyn = form.checkbox("Make the playlist on Spotify.")
submitted = form.form_submit_button("Make my playlist!")

if submitted:
    orig = int(origstr.split(" ... ")[0].split(" id: ")[1])
    dest = int(deststr.split(" ... ")[0].split(" id: ")[1])

    if orig == dest:
        st.write("Songs are the same! Try again.")
        submitted = False
    else:
        playlistDF = prodplay.makePlaylist(dataset, orig, dest, nsongs)
        st.write("## Here's your playlist!")

        chart = alt.Chart(playlistDF).mark_line(
            point=True, color="#D73F09"
        ).encode(
            x='valence', y='arousal',
            tooltip=['id-deezer', 'title', 'artist', 'valence', 'arousal']
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(playlistDF[["artist", "title", "valence", "arousal"]], use_container_width=True)

        if spotyn:
            title = "Stream Playlist {}".format(str(time.strftime("%Y-%m-%d-%H:%M")))
            spid = spotify.makePlaylist(sp, spo, title, playlistDF["id-spotify"], True)
            
            splink = "https://open.spotify.com/playlist/{}".format(spid)
            st.markdown("[Playlist on Spotify]({})".format(splink))
