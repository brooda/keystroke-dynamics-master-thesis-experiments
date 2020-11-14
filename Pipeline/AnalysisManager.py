import numpy as np
import matplotlib.pyplot as plt 
from .Constants import *

class AnalysisManager:
    def __init__(self):
        self.thresholds = np.linspace(1, 0, 26)
        self.results = []


    def __get_far_frr(self, prediction, real):
        tp = np.sum((prediction == 1) & (real == 1))
        tn = np.sum((prediction == 0) & (real == 0))
        fp = np.sum((prediction == 1) & (real == 0))
        fn = np.sum((prediction == 0) & (real == 1))
    
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        frr = 1-tpr

        return fpr, frr


    def save_far_frr(self, prediction, real):
        prediction = prediction[:, 1].reshape(1, -1)[0]

        fars = []
        frrs = []

        for threshold in self.thresholds:
            tmp_prediction = np.zeros(prediction.shape)
            tmp_prediction[prediction > threshold] = 1
            
            far, frr = self.__get_far_frr(tmp_prediction, real)
            fars.append(far)
            frrs.append(frr)

        self.results.append((np.array(fars), np.array(frrs)))


    def reset(self):
        self.results = []


    def calculate_frrs_at_given_fars(self, far, frr, given_fars):
        frr_s = []

        for given_far in given_fars:
            id_thres = np.argmax(far >= given_far)

            if given_far == 0.00:
                point_frr = frr[id_thres]
                frr_s.append(point_frr)
            else:
                far_1 = far[id_thres-1]
                far_2 = far[id_thres]
                frr_1 = frr[id_thres-1]
                frr_2 = frr[id_thres]
                
                searched_frr = self.find_frr_on_line(far_2, far_1, frr_2, frr_1, given_far)
                frr_s.append(searched_frr)

        return np.clip(frr_s, None, 1)


    def mean_frr(self, far_aggregate, frr_aggregate):
        frrs = []

        for far, frr in zip(far_aggregate, frr_aggregate):
            normalized_frr = self.calculate_frrs_at_given_fars(far, frr, np.arange(0, 1.01, 0.01))
            if not np.isnan(normalized_frr).any():
               frrs.append(normalized_frr)

        return np.mean(np.array(frrs), axis=0)


    def find_frr_on_line(self, far_2, far_1, frr_2, frr_1, point_far):
        del_far = far_2 - far_1
        del_frr = frr_2 - frr_1

        scale = point_far - far_1
        searched_frr = frr_1 + scale * del_frr / del_far
        return searched_frr


    def score(self, is_final):
        summary = []

        far_aggregate = []
        frr_aggregate = []
        frrs_at_given_fars_aggregate = []
        final_score_aggregate = []

        for far, frr in self.results:
            far_aggregate.append(far)
            frr_aggregate.append(frr)

            frrs_at_given_fars = self.calculate_frrs_at_given_fars(far, frr, [0.005, 0.01, 0.02, 0.03, 0.04])

            if not np.isnan(frrs_at_given_fars).any():
                frrs_at_given_fars_aggregate.append(frrs_at_given_fars)
                final_score = np.average(frrs_at_given_fars, weights=[5, 2, 2, 1, 1])
                final_score_aggregate.append(final_score)

        frrs_at_given_fars_aggregate = np.array(frrs_at_given_fars_aggregate)
        final_score_aggregate = np.array(final_score_aggregate)

        frrs_at_given_fars_mean = np.mean(frrs_at_given_fars_aggregate, axis=0)
        frrs_at_given_fars_std = np.std(frrs_at_given_fars_aggregate, axis=0)
        final_score_mean = np.mean(final_score_aggregate, axis=0)
        final_score_std = np.std(final_score_aggregate, axis=0)

        if not is_final:
            return {
                "final_score_mean": final_score_mean,
            }

        best_users_ind = np.argsort(final_score_aggregate)[: int(0.25 * len(self.results))]
        far_aggregate = np.array(far_aggregate)
        frr_aggregate = np.array(frr_aggregate)
        far_percentile_25 = far_aggregate[best_users_ind]
        frr_percentile_25 = frr_aggregate[best_users_ind]

        summary = {
            "far": np.arange(0, 1.01, 0.01),
            "frr": self.mean_frr(far_aggregate, frr_aggregate),
            "frr_percentile_25": self.mean_frr(far_percentile_25, frr_percentile_25),
            "frrs_at_given_fars_mean": frrs_at_given_fars_mean,
            "frrs_at_given_fars_std": frrs_at_given_fars_std,
            "final_score_mean": final_score_mean,
            "final_score_std": final_score_std,
        }

        return summary
        