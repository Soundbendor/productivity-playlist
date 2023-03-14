import pandas as pd
import json
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import helper
import segments

class SongDataset:
    def __init__(self, name, path, cols = None, feat_index = 5, arousal = 4, valence = 3, knn = False, verbose = False):
        self.name = name
        self.path = path
        self.cols = cols
        self.feat_index = feat_index
        self.arousal_index = arousal
        self.valence_index = valence

        self.full_df = pd.read_csv(path, header=0, index_col=0, usecols=cols).dropna()
        self.feat_df = self.full_df.iloc[:, feat_index:].copy()
        self.va_df = self.full_df.iloc[:, [valence, arousal]].copy()
        self.size = len(self.feat_df)
        self.verbose = verbose

        self.unique_points = None
        self.points_hash = None
        self.knn_model = None
        self.unique_size = None

        if verbose: 
            print("\n{}: dataset created".format(name))
        if knn:
            self.make_knn()
        if verbose:
            print()
        return
    
    def get_random_song(self, point):
        pstring = helper.arr2stringPoint(point)
        rand_idx = random.randint(0, len(self.points_hash[pstring])-1)
        song = self.points_hash[pstring][rand_idx]
        return song

    def get_all_songs(self, point):
        pstring = helper.arr2stringPoint(point)
        songs = self.points_hash[pstring]
        return songs
    
    def get_point(self, song):
        return self.va_df.loc[song]

    def get_feats(self, song, subset = ""):
        if subset == "" or subset == "head" or subset == "tail":
            return self.feat_df.loc[song]
        else:
            return self.feat_df[subset].loc[song]
    
    def get_spid(self, song):
        return self.full_df.loc[song]['sp_track_id']

    def make_unique(self):
        self.unique_points = []
        self.points_hash = {}
        if self.verbose: 
            print("{}: making unique points ... ".format(self.name), end='')

        for i in range(self.size):
            id = self.full_df.index.values[i]
            vector = self.va_df.iloc[i].tolist()
            self.unique_points.append(vector)
            
            vstring = helper.arr2stringPoint(vector)
            if vstring not in self.points_hash:
                self.points_hash[vstring] = []
            self.points_hash[vstring].append(id)

        self.unique_points = np.unique(np.array(self.unique_points), axis=0)
        self.unique_size = len(self.unique_points)
        
        if self.verbose:
            print("done!")
        return

    def make_knn(self):
        if self.unique_points is None:
            self.make_unique()

        if self.verbose:
            print("{}: making KNN model ... ".format(self.name), end='')
        
        self.knn_model = NearestNeighbors()
        self.knn_model.fit(self.unique_points)
    
        if self.verbose:
            print("done!")
        return self.knn_model
    
    def __len__(self):
        return self.size

class SegmentDataset(SongDataset):
    def __init__(self, name, path, cols = None, feat_index = 5, arousal = 4, valence = 3, knn = False, verbose = False):
        super().__init__(name, path, cols, feat_index, arousal, valence, knn, verbose)

        headkeys = segments.fillcols("head", {}).keys()
        if set(headkeys).issubset(set(self.feat_df.columns)):
            self.head_df = self.feat_df[headkeys]

        tailkeys = segments.fillcols("tail", {}).keys()
        if set(tailkeys).issubset(set(self.feat_df.columns)):
            self.tail_df = self.feat_df[tailkeys]

        return
    
    def get_feats(self, song, subset = ""):
        if subset == "head":
            return self.head_df.loc[song]
        elif subset == "tail":
            return self.tail_df.loc[song]
        elif subset == "":
            return self.feat_df.loc[song]
        else:
            return self.feat_df[subset].loc[song]


