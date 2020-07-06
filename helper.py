import matplotlib.pyplot as plt
import os
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

def sign(num):
    if num > 0:
        return '+'
    else:
        return '-'

def makeDir(key):
    if not os.path.exists(key):
        os.makedirs(key)

def string2arrPoint(key):
    positions = key.split()
    return [float(positions[0]), float(positions[1])]

def arr2stringPoint(arr):
    return "{}{} {}{}".format(sign(arr[0]), abs(arr[0]), sign(arr[1]), abs(arr[1]))

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
        if (line_count == 1):
            plt.plot(data[0], data[1], marker=marker, linestyle=linestyle)
        else:
            for i in range(line_count):
                plt.plot(data[i][0], data[i][1], marker=marker, linestyle=linestyle)
    
    if (legend != []):
        plt.legend(legend)

    if (title != ""):
        plt.title(title)

    if (file != ""):
        plt.savefig(file)

    plt.show(block=False)
    plt.clf()
