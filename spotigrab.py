import spotipy
import pprint
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np

import helper

#get important personal information from Spotify API
client_id = '2a285d92069147f8a7e59cec1d0d9bb6'
client_secret = '1eebc7035f74489db8f5597ce4afb863'
redirect_uri = 'https://www.google.com/'
username = 'eonkid46853'
scope = "user-library-read playlist-read-private"
sp = helper.Spotify(client_id, client_secret, redirect_uri, username, scope)

songdata = pd.read_csv("deezer-data/all2.csv", header=0, index_col=0)
song_ids = list(songdata.index.values)

# merge the artist and track to get a single search item
joined_titles = []

# make lists to store the relevant track features
sp_track_id = []
sp_acousticness = []
sp_danceability = []
sp_duration_ms = []
sp_energy = []
sp_instrumentalness = []
sp_key = []
sp_liveness = []
sp_loudness = []
sp_mode = []
sp_speechiness = []
sp_tempo = []
sp_time_sig = []
sp_valence = []
sp_popularity = []
sp_explicit = []
sp_artist_genres = {}

print(len(songdata))

for i in range(len(songdata)):
    print(i, end = "\r")
    joined_titles.append(songdata.iloc[i][4] + " " + songdata.iloc[i][5])
    result = sp.search(joined_titles[i], limit=1, type='track')
    # pprint.pprint(result)

    try:
        uri = result['tracks']['items'][0]['uri']
        popularity = result['tracks']['items'][0]['popularity']
        explicit = result['tracks']['items'][0]['explicit']
        artist_id = result['tracks']['items'][0]['artists'][0]['id']

        artist = sp.artist(artist_id)
        sp_artist_genres[song_ids[i]] = np.array(artist['genres'])
        features = sp.audio_features([uri])[0]
        
        sp_track_id.append(features['id'])
        sp_acousticness.append(features['acousticness'])
        sp_danceability.append(features['danceability'])
        sp_duration_ms.append(features['duration_ms'])
        sp_energy.append(features['energy'])
        sp_instrumentalness.append(features['instrumentalness'])
        sp_key.append(features['key'])
        sp_liveness.append(features['liveness'])
        sp_loudness.append(features['loudness'])
        sp_mode.append(features['mode'])
        sp_speechiness.append(features['speechiness'])
        sp_tempo.append(features['tempo'])
        sp_time_sig.append(features['time_signature'])
        sp_valence.append(features['valence'])
        sp_popularity.append(popularity)
        sp_explicit.append(explicit)


    except:
        sp_track_id.append(None)
        sp_acousticness.append(None)
        sp_danceability.append(None)
        sp_duration_ms.append(None)
        sp_energy.append(None)
        sp_instrumentalness.append(None)
        sp_key.append(None)
        sp_liveness.append(None)
        sp_loudness.append(None)
        sp_mode.append(None)
        sp_speechiness.append(None)
        sp_tempo.append(None)
        sp_time_sig.append(None)
        sp_valence.append(None)
        sp_popularity.append(None)
        sp_explicit.append(None)

genre_df = pd.DataFrame.from_dict(sp_artist_genres, orient='index')
pprint.pprint(genre_df)

songdata['sp_track_id'] = sp_track_id
songdata['sp_acousticness'] = sp_acousticness
songdata['sp_danceability'] = sp_danceability
songdata['sp_duration_ms'] = sp_duration_ms
songdata['sp_energy'] = sp_energy
songdata['sp_instrumentalness'] = sp_instrumentalness
songdata['sp_key'] = sp_key
songdata['sp_liveness'] = sp_liveness
songdata['sp_loudness'] = sp_loudness
songdata['sp_mode'] = sp_mode
songdata['sp_speechiness'] = sp_speechiness
songdata['sp_tempo'] = sp_tempo
songdata['sp_time_sig'] = sp_time_sig
songdata['sp_valence'] = sp_valence
songdata['sp_popularity'] = sp_popularity
songdata['sp_explicit'] = sp_explicit

songdata.to_csv(path_or_buf='all2-spotify.csv')