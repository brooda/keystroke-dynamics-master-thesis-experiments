import numpy as np
from .Unpacker import Unpacker
from .Cache import Cache
import hashlib

class ToBinaryTransformator:
    def __init__(self, train, test, args, zero_class_multiplier_for_train = 5, zero_class_multiplier_for_test = 2):
        self.train = train
        self.test = test
        
        np.random.seed(0)

        self.classes = np.unique(train.user)
        self.classes_num = np.size(self.classes)
        self.current_class_num = 0

        self.zero_class_multiplier_for_train = zero_class_multiplier_for_train
        self.zero_class_multiplier_for_test = zero_class_multiplier_for_test

        self.precomputed_sets = []

        cache = Cache(f"cache.p")

        while self.has_next():
            tmp_train, tmp_test = self.yield_next_train_test()
            self.data_unpacker = Unpacker(tmp_train, tmp_test, args.used_data, cache, self.current_class_num)

            x_train, y_train = self.data_unpacker.unpack(train=True)
            x_test, y_test = self.data_unpacker.unpack(train=False)
            self.precomputed_sets.append((x_train, y_train, x_test, y_test))
        
        print("len(self.precomputed_sets", len(self.precomputed_sets))
        self.reload()
        print("ToBinaryTransformator created and data loaded")

    def get_next(self):
        ret = self.precomputed_sets[self.current_class_num]
        self.current_class_num += 1
        return ret


    def reload(self):
        self.current_class_num = 0


    def has_next(self):
        return self.current_class_num < self.classes_num 


    def __to_binary(self, df, target_class, zero_class_multiplier):
        ones_indices = np.where(df.user == target_class)[0]
        df.user = 0
        df.user[ones_indices] = 1
        
        class_cardinality = np.sum(df.user)
        non_class_cardinality = zero_class_multiplier * class_cardinality

        zeros_indices = np.random.choice(np.where(df.user == 0)[0], non_class_cardinality)

        needed_indices = np.concatenate([zeros_indices, ones_indices])

        df = df.loc[needed_indices, :]
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df


    def yield_next_train_test(self):
        target_class = self.classes[self.current_class_num]
        self.current_class_num += 1

        tmp_train = self.train.copy()
        tmp_test = self.test.copy()

        tmp_train = self.__to_binary(tmp_train, target_class, self.zero_class_multiplier_for_train)
        tmp_test = self.__to_binary(tmp_test, target_class, self.zero_class_multiplier_for_test)

        return tmp_train, tmp_test
