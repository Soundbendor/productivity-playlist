import pandas as pd
import json
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import helper
from songdataset import SongDataset
import segments

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





