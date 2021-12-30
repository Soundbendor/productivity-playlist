import pandas as pd
import json
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import helper

class SongDataset:
    def __init__(self, name, path, cols, start_index, spotify=False):
        self.name = name
        self.spotify = spotify
        self.full_df = pd.read_csv(path, header=0, index_col=0, usecols=cols).dropna()
        self.data_df = self.full_df.iloc[:, start_index:].copy()
        self.size = len(self.data_df)

        self.unique_points = None
        self.points_hash = None
        self.knn_model = None
        self.unique_size = None
        return
    
    def get_song(self, point):
        pstring = helper.arr2stringPoint(point)
        rand_idx = random.randint(0, len(self.points_hash[pstring])-1)
        song = self.points_hash[pstring][rand_idx]
        return song
    
    def get_point(self, song):
        return self.data_df.loc[song]
    
    def get_spid(self, song):
        return self.full_df.loc[song]['sp_id']

    def make_unique(self):
        self.unique_points = []
        self.points_hash = {}

        for i in range(self.size):
            id = self.full_df.index.values[i]
            vector = self.data_df.iloc[i].tolist()
            self.unique_points.append(vector)
            
            vstring = helper.arr2stringPoint(vector)
            if vstring not in self.points_hash:
                self.points_hash[vstring] = []
            self.points_hash[vstring].append(id)

        self.unique_points = np.unique(np.array(self.unique_points), axis=0)
        self.unique_size = len(self.unique_points)
        return

    def make_knn(self):
        if self.unique_points is None:
            self.make_unique()
        
        self.knn_model = NearestNeighbors()
        self.knn_model.fit(self.unique_points)
        return self.knn_model
    
    def __len__(self):
        return self.size







