import spotipy
import pprint
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np

#get important personal information from Spotify API
client_id = '2a285d92069147f8a7e59cec1d0d9bb6'
client_secret = '1eebc7035f74489db8f5597ce4afb863'
redirect_uri = 'https://www.google.com/'
username = 'eonkid46853'

#get yo Spotify
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
scope = 'user-library-read playlist-read-private'
try:
    token = util.prompt_for_user_token(username, scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
    sp = spotipy.Spotify(auth= token)
except:
    print('Token is not accessible for ' + username)

songdata = pd.read_csv("deezer-data\\train.csv", header=0, index_col=0, usecols=[0, 3, 4, 5, 6])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
print(songdata.iloc[0])

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

print(len(songdata))

for i in range(len(songdata)):
    print(i, end="\r")
    joined_titles.append(songdata.iloc[i][2] + " " + songdata.iloc[i][3])
    result = sp.search(joined_titles[i], limit=1, type='track')

    try:
        uri = result['tracks']['items'][0]['uri']
        features = sp.audio_features(uri)[0]
        
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

    except:
        sp_track_id.append('NO TRACK FOUND ON SPOTIFY')
        sp_acousticness.append(0)
        sp_danceability.append(0)
        sp_duration_ms.append(0)
        sp_energy.append(0)
        sp_instrumentalness.append(0)
        sp_key.append(0)
        sp_liveness.append(0)
        sp_loudness.append(0)
        sp_mode.append(0)
        sp_speechiness.append(0)
        sp_tempo.append(0)
        sp_time_sig.append(0)
        sp_valence.append(0)       

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

songdata.to_csv(path_or_buf='data_with_features.csv')
