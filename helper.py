import matplotlib.pyplot as plt

import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

def Spotify(client_id, client_secret, redirect_uri, username, scope):
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    try:
        token = util.prompt_for_user_token(username, scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri)
        sp = spotipy.Spotify(auth= token)
    except:
        print('Token is not accessible for ' + username)

    sp = spotipy.Spotify(auth=token, client_credentials_manager=client_credentials_manager)    
    return sp

def makeSpotifyList(sp, username, title, track_ids, public = False):
    result_playlist = sp.user_playlist_create(username, title, public=public)
    sp.user_playlist_add_tracks(username, result_playlist['id'], track_ids)

def graph(xlabel, ylabel, data, data_dim = 1, line_count = 1):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if (data_dim == 1):
        plt.plot(data)
    
    elif (data_dim == 2):
        if (line_count > 1):
            for i in range(line_count):
                plt.plot(data[i][0], data[i][1])
        else:
            plt.plot(data[0], data[1])
    
    plt.show()
