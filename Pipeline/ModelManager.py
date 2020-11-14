import numpy as np
from .Constants import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.svm
from sklearn.neural_network import MLPClassifier

def get_model_by_name(name, model_parameter):
    if name == KNN:
        return KNN_model(model_parameter)
    if name == SVC:
        return SVC_model(model_parameter)
    if name == NEURAL_NETWORK:
        return Perceptron_model(model_parameter)
    if name == RANDOM_FOREST:
        return RandomForest_model(model_parameter)

def get_parameters_for_model(name):
    ret = []
    
    if name == KNN:
        for metric in ["euclidean", "manhattan"]:
            for k in [3, 4, 5, 6, 7, 8, 9]:
                ret.append((metric, k))
    if name == SVC:
        ret = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    if name == NEURAL_NETWORK:
        ret = [(10, 10), (20, 20), (40, 40), (10, 10, 10), (20, 20, 20), (40, 40, 40)]
    if name == RANDOM_FOREST:
        ret = [5, 10, 20, 50, 100]

    return ret


def KNN_model(metric_and_k):
    metric, k = metric_and_k
    return KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')

def SVC_model(gamma):
    return sklearn.svm.SVC(kernel='rbf', gamma=gamma, class_weight={1: 10}, probability=True)

def RandomForest_model(n_estimators):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=0)

def Perceptron_model(hidden_layer_sizes):
    return MLPClassifier(solver='adam', hidden_layer_sizes=(hidden_layer_sizes), random_state=1, max_iter=200)