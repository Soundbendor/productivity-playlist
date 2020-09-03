import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
    fig = plt.gcf()
    ax = fig.gca()

    # add formatted labels
    titleFont = fm.FontProperties(fname="./fonts/Stratum2-Bold.otf")
    axisFont = fm.FontProperties(fname="./fonts/Stratum2-Medium.otf")
    legendFont = fm.FontProperties(fname="./fonts/KievitOffc-Ita.ttf")

    ax.set_xlabel(xlabel, fontproperties=axisFont, size=12)
    ax.set_ylabel(ylabel, fontproperties=axisFont, size=12)
    ax.set_title(title, fontproperties=titleFont, size=32)

    if (data_dim == 1):
        for i in range(line_count):
            plt.plot(data[i], marker=marker, linestyle=linestyle)
    
    elif (data_dim == 2):
        if (line_count == 1):
            plt.plot(data[0], data[1], marker=marker, color="#D73F09", linestyle=linestyle)
        else:
            for i in range(line_count):
                plt.plot(data[i][0], data[i][1], marker=marker, linestyle=linestyle)
    
    if (legend != []):
        ax.legend(legend, prop=legendFont)

    if (title != ""):
        plt.title(title)

    if (file != ""):
        plt.savefig(file, dpi=300)

    plt.show(block=False)
    plt.clf()

def plot_AV_data(x, y, title="", colors="#D73F09", file="./test.png", plt_size=10):
    plt.figure(figsize=(plt_size,plt_size))
    plt.scatter(x, y, s=20, c=colors, alpha=.5)
    plt.xlim(-1.25,1.25)
    plt.ylim(-1.25,1.25)  

    # draw the unit circle
    fig = plt.gcf()
    ax = fig.gca()
    circle1 = plt.Circle((0, 0), 1.0, color='0.25', fill=False)
    ax.add_artist(circle1)

    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_position(("data",0))
    ax.spines["right"].set_position(("data",0))

    # add formatted labels
    titleFont = fm.FontProperties(fname="./fonts/Stratum2-Bold.otf")
    axisFont = fm.FontProperties(fname="./fonts/Stratum2-Medium.otf")
    emotionFont = fm.FontProperties(fname="./fonts/KievitOffc-BoldIta.ttf")

    ax.set_xlabel("Valence", fontproperties=axisFont, size=plt_size*2)
    ax.set_ylabel("Arousal", fontproperties=axisFont, size=plt_size*2)
    ax.set_title(title, fontproperties=titleFont, size=plt_size*3)

    # print emotion labels
    ax.text(0.98, 0.35, 'Happy', fontproperties=emotionFont, size=int(plt_size*1.5))
    ax.text(0.5, 0.9, 'Excited', fontproperties=emotionFont, size=int(plt_size*1.5))
    ax.text(-1.16, 0.35, 'Afraid', fontproperties=emotionFont, size=int(plt_size*1.5))
    ax.text(-0.7, 0.9, 'Angry', fontproperties=emotionFont, size=int(plt_size*1.5))
    ax.text(-1.13, -0.25, 'Sad', fontproperties=emotionFont, size=int(plt_size*1.5))
    ax.text(-0.9, -0.9, 'Depressed', fontproperties=emotionFont, size=int(plt_size*1.5))
    ax.text(0.98, -0.25, 'Content', fontproperties=emotionFont, size=int(plt_size*1.5))
    ax.text(0.7, -0.9, 'Calm', fontproperties=emotionFont, size=int(plt_size*1.5)) 

    plt.savefig(file, dpi=300)
    plt.show(block=False)
    plt.clf() 
