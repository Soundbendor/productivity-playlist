import spotipy
import spotipy.util as util
import spotipy.oauth2 as oauth

def Spotify(client_id, client_secret, redirect_uri, username, scope, auto=False):
    cache_path='.cache-{}'.format(username)
    spo = oauth.SpotifyOAuth(
        client_id = client_id,
        client_secret = client_secret,
        redirect_uri = redirect_uri,
        scope = scope,
        cache_path=cache_path
    )
    sp = spotipy.Spotify(auth_manager=spo)
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
