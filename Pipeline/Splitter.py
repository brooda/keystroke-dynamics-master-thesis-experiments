import pandas as pd
import numpy as np

class Splitter:
    def __init__(self, data_path, train_sessions, test_sessions, repetitions):
        data = pd.read_csv(data_path)
        
        self.train = data[data.session.isin(train_sessions) & data.repetition.between(repetitions[0], repetitions[1])].reset_index(drop=True)
        self.test = data[data.session.isin(test_sessions) & data.repetition.between(repetitions[0], repetitions[1])].reset_index(drop=True)


    def get_train(self):
        return self.train
        
    def get_test(self):
        return self.test