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

songtitles = pd.read_csv("deam-data\\metadata\\metadata_2015.csv", header=0, usecols=[2, 3])
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
print(songtitles.iloc[0])

# merge the artist and track to get a single search item
joined_titles = []

for i in range(len(songtitles)):
    joined_titles.append(songtitles.iloc[i][0] + " " + songtitles.iloc[i][1])

result = sp.search(joined_titles[0], limit=1, type='track')
features = sp.audio_features(result)
pprint.pprint(result)
pprint.pprint(features)