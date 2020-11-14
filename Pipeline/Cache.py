import pickle
import os

class Cache:
    def __init__(self, path=""):
        self.path = path
        
        if os.path.exists(self.path):
                with open(self.path, 'rb') as handle:
                    self.modification_times = pickle.load(handle)
        else:
            self.modification_times = {}
            

    def has(self, element):
        return element in self.modification_times


    def get(self, key):
        return self.modification_times[key]


    def save(self, key, value):
        self.modification_times[key] = value


    def save_dump(self):
        with open(self.path, 'wb') as handle:
            pickle.dump(self.modification_times, handle, protocol=pickle.HIGHEST_PROTOCOL)
    