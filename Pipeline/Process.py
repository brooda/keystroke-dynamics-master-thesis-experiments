from .Splitter import Splitter
from .ToBinaryTransformator import ToBinaryTransformator
from .Unpacker import Unpacker
from .AnalysisManager import AnalysisManager
from .ModelManager import *
from .Constants import *
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import warnings
# TODO: refactor
warnings.filterwarnings('ignore')

class Process:
    def __init__(self, args):
        self.args = args

        self.splitter = Splitter(args.data_path, train_sessions = args.train_sessions, test_sessions = args.test_sessions, repetitions = args.repetitions)
        train = self.splitter.get_train()
        test = self.splitter.get_test()

        self.binary_tasks_generator = ToBinaryTransformator(train, test, self.args, 1, 1)
        self.models = args.models

        # used for parameter tuning
        self.local_analysis_manager = AnalysisManager()
        # used for global score
        self.global_analysis_manager = AnalysisManager()
    

    def run(self):
        results = {}

        for model_name in self.models:
            self.global_analysis_manager.reset()
            model_results = {}
            best_model_parameters = []

            self.binary_tasks_generator.reload()
            while self.binary_tasks_generator.has_next():
                x_train, y_train, x_test, y_test = self.binary_tasks_generator.get_next()

                best_score = 2
                best_parameter = None
                
                # parameter selection for model
                for parameter in get_parameters_for_model(model_name):
                    model = get_model_by_name(model_name, parameter)
                    self.pipe = Pipeline([('scaler', StandardScaler()), (model_name, model)])
                    
                    self.local_analysis_manager.reset()

                    kf = KFold(n_splits=4)
                    for train_index, validation_index in kf.split(x_train):
                        tmp_x_train, tmp_x_validation = x_train[train_index], x_train[validation_index]
                        tmp_y_train, tmp_y_validation = y_train[train_index], y_train[validation_index]

                        try:
                            self.pipe.fit(tmp_x_train, tmp_y_train);
                            prediction = self.pipe.predict_proba(tmp_x_validation)
                            self.local_analysis_manager.save_far_frr(prediction, tmp_y_validation)
                        except:
                            print("cos sie popsulo")
                            raise

                    score = self.local_analysis_manager.score(False)["final_score_mean"]
                    if score < best_score:
                        best_score = score
                        best_parameter = parameter
            
                best_model_parameters.append(best_parameter)
                model = get_model_by_name(model_name, best_parameter)
                self.pipe = Pipeline([('scaler', StandardScaler()), (model_name, model)])

                self.pipe.fit(x_train, y_train)
                prediction = self.pipe.predict_proba(x_test)
                self.global_analysis_manager.save_far_frr(prediction, y_test)
            sc = self.global_analysis_manager.score(True)
            model_results["scores"] = sc
            model_results["best_parameters"] = best_model_parameters
            
            results[model_name] = model_results

        return results