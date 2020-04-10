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

def graph(xlabel, ylabel, data, data_dim = 1, line_count = 1, legend = [], file = "", marker=',', linestyle='-', title=""):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if (data_dim == 1):
        for i in range(line_count):
            plt.plot(data[i], marker=marker, linestyle=linestyle)
    
    elif (data_dim == 2):
        for i in range(line_count):
            plt.plot(data[i][0], data[i][1], marker=marker, linestyle=linestyle)
    
    if (legend != []):
        plt.legend(legend)

    if (title != []):
        plt.title(title)

    if (file != ""):
        plt.savefig(file)

    plt.show()
