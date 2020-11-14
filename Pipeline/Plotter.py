import numpy as np
import matplotlib.pyplot as plt 
from Constants import *

class AnalysisManager:
    def __init__(self, options, graph_path = "./graphs"):
        self.graph_path = graph_path
        self.options = options
        self.results = []

    def __get_far_frr(self, prediction, real):
        tp = np.sum((prediction == 1) & (real == 1))
        tn = np.sum((prediction == 0) & (real == 0))
        fp = np.sum((prediction == 1) & (real == 0))
        fn = np.sum((prediction == 0) & (real == 1))
    
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)

        return fpr, 1-tpr

    def analyze(self, prediction, real):
        prediction = prediction[:, 1].reshape(1, -1)[0]

        if COUNT_RECALL_PRECISION_IN_THRESHOLDS in self.options:
            recalls = []
            precisions = []

            for threshold in np.linspace(1, 0, 25):
                tmp_prediction = np.zeros(prediction.shape)
                tmp_prediction[prediction > threshold] = 1
                
                rec, pre = self.__get_far_frr(tmp_prediction, real)
                recalls.append(rec)
                precisions.append(pre)

            self.results.append((recalls, precisions))

        elif COUNT_SINGLE_RECALL_PRECISION in self.options:
            recall, precision = self.__get_far_frr(prediction, real)
            self.results.append((recall, precision))


    def __plot_far_frr(self, far, frr):
        plt.rcParams["font.size"] = "15"
        plt.plot(far, frr)
        plt.xlabel("FAR")
        plt.ylabel("FRR")


        far_s = [0.00, 0.01, 0.02, 0.05, 0.10]
        frr_s = []

        for point_far in far_s:
            id_thres = np.argmax(far > point_far)

            # ZM-FAR
            if point_far == 0.00:
                point_frr = frr[id_thres]
                frr_s.append(point_frr)
                plt.scatter(point_far, point_frr, c="r")
                plt.annotate(f"{point_far}, {round(point_frr, 2)}", (point_far, point_frr))
            else:
                far_1 = far[id_thres-1]
                far_2 = far[id_thres]
                frr_1 = frr[id_thres-1]
                frr_2 = frr[id_thres]
                
                del_far = far_2 - far_1
                del_frr = frr_2 - frr_1

                scale = point_far - far_1
                point_frr = frr_1 + scale * del_frr / del_far
                
                frr_s.append(point_frr)
                plt.scatter(point_far, point_frr, c="r")


                plt.annotate(f"{point_far}, {round(point_frr, 2)}", (point_far, point_frr))
                

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig("far_frr.png")

        return frr_s

    def __scatter_far_frr(self, far, frr):
        plt.scatter(far, frr)
        plt.xlabel("FAR")
        plt.ylabel("FRR ")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig("recall-precision.png")


    def summarize(self):
        results = np.array(self.results)

        if COUNT_RECALL_PRECISION_IN_THRESHOLDS in self.options:
            # recall - precision graph
            results = np.mean(results, axis=0)
            return self.__plot_far_frr(results[0], results[1])
        elif COUNT_SINGLE_RECALL_PRECISION in self.options:
            mn = np.mean(results, axis=0)
            return np.round(mn, 2)