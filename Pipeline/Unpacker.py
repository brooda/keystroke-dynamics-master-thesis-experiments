import numpy as np
import pandas as pd
import json
from .Constants import *
from .Toeplitz import toeplitz_coefficients
from .dtw import dtw_distance

from scipy.spatial.distance import euclidean

import multiprocessing as mp
import hashlib

class DTWDistanceCounter(object):
    # by_fragments - True will mean that base_sequence will be a collection of arrays
    def __init__(self, base_sequence, by_fragments = False):
        self.base_sequence = base_sequence
        self.by_fragments = by_fragments
    def __call__(self, compared_sequence):
        def dist(s1, s2):
            l1 = len(s1)
            l2 = len(s2)
            div = max(l1, l2)
            
            if l1 == 0 and l2 == 0:
                return 0;

            if l1 == 0:
                return sum(s2) / l2

            if l2 == 0:
                return sum(s1) / l1

            return dtw_distance(s1, s2) / div

        if self.by_fragments:
            dists = [dist(base_fragment, compared_fragment) for base_fragment, compared_fragment in zip(self.base_sequence, compared_sequence)]
            return dists
        else:
            return dist(self.base_sequence, compared_sequence)


class Unpacker:
    def __init__(self, train_df, test_df, data_type_used, cache, current_class_num):
        self.data_type_used = data_type_used

        self.train_df = train_df
        self.positive_train = train_df[train_df.user == 1]

        self.test_df = test_df
        self.cache = cache

        self.current_class_num = current_class_num


    def unpack(self, train):
        if train:
            df = self.train_df
        else:
            df = self.test_df
        
        if "user" not in df.columns:
            raise Exception("No user column in dataframe")


        def load_json(x):
            return json.loads(x)     

        def dtw_distances(sequence, data_type_to_transform, by_fragments = False):
            hashId = hashlib.md5()
            data = {"seq":sequence, "data_type": data_type_to_transform[0]}
            hashId.update(repr(data).encode('utf-8'))
            data_key = hashId.hexdigest()

            if self.cache.has(data_key):
                distances = self.cache.get(data_key)
            else:
                pool = mp.Pool(mp.cpu_count())
                distances = pool.map(DTWDistanceCounter(sequence, by_fragments), [load_json(row_in_train[data_type_to_transform[0]]) for _, row_in_train in self.positive_train.iterrows()])
                pool.close()        


                if by_fragments:
                    distances = np.array(distances)
                    exclude = np.argmin(np.sum(distances, axis=1))
                    mask = np.ones(distances.shape, bool)
                    mask[exclude,] = False
                    distances = distances[mask]
                    distances = list(distances.flatten())
                else:
                    minimal = min(distances)
                    distances.remove(minimal)
                    distances.append(min(distances))

                self.cache.save(data_key, distances)

            return distances

        # For each row, merged list will contain full feature vector (each in form of 1-dimensional list)
        # Finally list of lists will be transformed into 2d array)
        def create_feature_vector(row):
            features_coefficients = []

            for transformation, data_type_to_transform in self.data_type_used:
                if transformation == NO_TRANSFORMATION:
                    if len(data_type_to_transform) > 1:
                        raise Exception("NO_TRANFORMATION accepts only one argument")
                    arr = load_json(row[data_type_to_transform[0]])
                    features_coefficients.append(arr)

                elif transformation == TOEPLITZ_TRANSFORMATION:
                    toeplitz_args = []
                    
                    for data_type in data_type_to_transform:
                        toeplitz_args.append(load_json(row[data_type]))

                    toeplitz_res = toeplitz_coefficients(toeplitz_args)
                    features_coefficients.append(toeplitz_res)

                elif transformation == DIFF_TRANSFORMATION:
                    if len(data_type_to_transform) > 1:
                        raise Exception("DIFF_TRANFORMATION accepts only one argument")
                    ret = np.diff(load_json(row[data_type_to_transform[0]]))
                    features_coefficients.append(list(ret))

                elif transformation == SUM:
                    if len(data_type_to_transform) > 1:
                        raise Exception("SUM accepts only one argument")
                    
                    features_coefficients.append([np.sum(load_json(row[data_type_to_transform[0]]))])
                    features_coefficients.append([np.std(load_json(row[data_type_to_transform[0]]))])

                elif transformation == DTW_DISTANCES:
                    if len(data_type_to_transform) > 1:
                        raise Exception("DTW_DISTANCES accepts only one argument")

                    # we will count distance of this sequence from all positive train sequences
                    sequence = load_json(row[data_type_to_transform[0]])
                    distances = dtw_distances(sequence, data_type_to_transform)
                    features_coefficients.append(distances)
                
                elif transformation == DTW_DISTANCES_PARTS:
                    sequence = load_json(row[data_type_to_transform[0]])
                    distances = dtw_distances(sequence, data_type_to_transform, True)
                    features_coefficients.append(distances)

            # a list of items is returned
            ret = np.concatenate(features_coefficients).ravel()
            return ret

        df["features_coefficients"] = df.apply(create_feature_vector, axis=1)
        coeffs = np.array(list(df["features_coefficients"]))
       
        return coeffs, df.user.to_numpy()
