import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import spotipy
import spotipy.util as util
import spotipy.oauth2 as oauth
import sys
import json
import numpy as np
import pprint
import re

def loadConfig(configFile=None):
    info = {}
    if configFile is None:
        configFile = sys.argv[1] if (len(sys.argv) > 1) else input("Please enter the path of your config file: ")
    while not os.path.exists(configFile) or not configFile.endswith(".json"):
        configFile = input("Config JSON file not found! Please enter a valid path: ")
    with open(configFile) as f:
        info = json.load(f)
    return info

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
    return [float(positions[i]) for i in range(len(positions))]

def arr2stringPoint(arr):
    s = ""
    for i in range(len(arr)):
        s = s + "{}{} ".format(sign(arr[i]), abs(arr[i]))
    return s[:-1]

def Spotify(client_id, client_secret, redirect_uri, username, scope, auto=False):
    spo = oauth.SpotifyOAuth(
        client_id = client_id,
        client_secret = client_secret,
        redirect_uri = redirect_uri,
        scope = scope,
        cache_path = '.cache-{}'.format(username)
    )
    client_credentials_manager = oauth.SpotifyClientCredentials(
        client_id=client_id, 
        client_secret=client_secret
    )
    try:
        if (auto):
            token = util.prompt_for_user_token(
                username, 
                scope, 
                client_id=client_id, 
                client_secret=client_secret, 
                redirect_uri=redirect_uri
            )
        else:
            _auth_finder = re.compile("code=(.*?)$", re.MULTILINE)
            auth = spo.get_authorize_url()
            print(auth)
            auth_url = input('Click the link above and copy and paste the url here: ')
            _re_auth = re.findall(_auth_finder, auth_url)
            token = spo.get_access_token(_re_auth[0])
            token2 = util.prompt_for_user_token(
                username, 
                scope, 
                client_id=client_id, 
                client_secret=client_secret, 
                redirect_uri=redirect_uri
            )
        
        sp = spotipy.Spotify(auth=token['access_token'])
    except:
        print('Token is not accessible for ' + username)

    sp = spotipy.Spotify(auth=token['access_token'], client_credentials_manager=client_credentials_manager)    
    return sp, spo

def refresh_token(spo):
    cached_token = spo.get_cached_token()
    refreshed_token = cached_token['refresh_token']
    new_token = spo.refresh_access_token(refreshed_token)
    # also we need to specifically pass `auth=new_token['access_token']`
    sp = spotipy.Spotify(auth=new_token['access_token'])
    return sp

def makeSpotifyList(sp, spo, title, track_ids, public = False):
    try:
        result_playlist = sp.user_playlist_create(sp.me()["id"], title, public=public)
    except spotipy.client.SpotifyException:
        sp = refresh_token(spo)
        result_playlist = sp.user_playlist_create(sp.me()["id"], title, public=public)
    
    sp.user_playlist_add_tracks(sp.me()["id"], result_playlist['id'], track_ids)
    return result_playlist['id']

def graph(xlabel, ylabel, data, data_dim = 1, line_count = 1, legend = [], file = "", marker=',', linestyle='-', title="", unit_size=2, width=6.4, height=4.8, hist=False):
    plt.figure(figsize=(width*unit_size,height*unit_size))
    fig, ax= plt.subplots(dpi=600)

    # add formatted labels
    titleFont = fm.FontProperties(fname="./static/fonts/KievitOffc-Bold.ttf",size='x-large')
    axisFont = fm.FontProperties(fname="./static/fonts/KievitOffc.ttf",size='x-large')
    legendFont = fm.FontProperties(fname="./static/fonts/KievitOffc-Ita.ttf",size='x-large')

    ax.set_xlabel(xlabel, fontproperties=axisFont)
    ax.set_ylabel(ylabel, fontproperties=axisFont)
    ax.set_title(title, fontproperties=titleFont)

    if not hist:
        if (data_dim == 1):
            for i in range(line_count):
                ax.plot(data[i], marker=marker, linestyle=linestyle)
        
        elif (data_dim == 2):
            if (line_count == 1):
                ax.plot(data[0], data[1], marker=marker, color="#D73F09", linestyle=linestyle)
            else:
                for i in range(line_count):
                    ax.plot(data[i][0], data[i][1], marker=marker, linestyle=linestyle)
    else:
        n, bins, patches = ax.hist(data, bins=int(1+3.3*np.log10(len(data))), facecolor="#D73F09")
    
    if (legend != []):
        ax.legend(legend, prop=legendFont)

    if (title != ""):
        plt.title(title)

    if (file != ""):
        plt.savefig(file, dpi=600)

    plt.show(block=False)
    plt.clf()

def plot_AV_box(plots, labels, title="test", file="./test.png", plt_size=10, vert=True, showfliers=True):
    plt.figure(figsize=(plt_size, plt_size))
    fig, ax= plt.subplots(dpi=600)
    
    titleFont = fm.FontProperties(fname="./static/fonts/Stratum2-Bold.otf")
    ax.set_title(title, fontproperties=titleFont)    
    
    plt.boxplot(plots, labels=labels, showmeans=True, meanline=True, vert=vert, showfliers=showfliers)
    plt.tight_layout()
    
    plt.savefig(file, dpi=600)
    plt.show(block=False)
    plt.clf() 

def plot_AV_data(v, a, title="", colors="#D73F09", file="./test.png", plt_size=10, alpha=.5):
    plt.figure(figsize=(plt_size,plt_size))
    plt.scatter(v, a, s=20, c=colors, alpha=alpha)
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
    titleFont = fm.FontProperties(fname="./static/fonts/Stratum2-Bold.otf")
    axisFont = fm.FontProperties(fname="./static/fonts/Stratum2-Medium.otf")
    emotionFont = fm.FontProperties(fname="./static/fonts/KievitOffc-BoldIta.ttf", size='xx-large')

    ax.set_xlabel("Valence", fontproperties=axisFont, size=plt_size*3)
    ax.set_ylabel("Arousal", fontproperties=axisFont, size=plt_size*3)
    ax.set_title(title, fontproperties=titleFont, size=plt_size*4)
    # ax.axes.xaxis.set_ticks([])
    # ax.axes.yaxis.set_ticks([])
    
    # print emotion labels
    ax.text(0.98, 0.35, 'Happy', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(0.5, 0.9, 'Excited', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(-1.16, 0.35, 'Afraid', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(-0.7, 0.9, 'Angry', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(-1.13, -0.25, 'Sad', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(-0.9, -0.9, 'Depressed', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(0.98, -0.25, 'Content', fontproperties=emotionFont, size=int(plt_size*2.5))
    ax.text(0.7, -0.9, 'Calm', fontproperties=emotionFont, size=int(plt_size*2.5)) 

    plt.savefig(file, dpi=600)
    plt.show(block=False)
    plt.clf() 
