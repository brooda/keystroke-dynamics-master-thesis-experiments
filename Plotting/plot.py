import pickle
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt 

plt.rcParams["font.size"] = "10"

def plot(file):
    with open(f'../ExperimentResults/{file}.p', 'rb') as handle:
        dictionary = pickle.load(handle)
        
    table_latex_string = ""
    #"\hline\n"
    #table_latex_string += "& score & 0.005 & 0.01 & 0.02 & 0.03 & 0.04 \\\\ \hline \n"
    
    
    for alg, dump in dictionary.items():
        scores = dump["scores"]
        

        far = scores["far"]
        frr = scores["frr"]
        
        far_25 = scores["far"]
        frr_25 = scores["frr_percentile_25"]
        
        plt.set_cmap('jet')
        plt.rcParams['lines.linewidth'] = 3.0
        plt.rcParams['axes.prop_cycle'] = cycler(color='bbggrryy')
        
        plt.plot(far, frr, label=f'{alg}')
        plt.plot(far_25, frr_25, '--', label=f'{alg}_percentile')
        coefficients_mean = np.concatenate([np.array([scores["final_score_mean"]]), scores["frrs_at_given_fars_mean"]])
        coefficients_std  = np.concatenate([np.array([scores["final_score_std"]]), scores["frrs_at_given_fars_std"]])

        #table_latex_string += f"\\textbf{{{alg}}} & "
        table_latex_string += "{"
        
        for m, s in zip(coefficients_mean, coefficients_std):
            table_latex_string += "{:.2f} ({:.2f}) & ".format(m, s)

        table_latex_string = table_latex_string[:-2]
        table_latex_string += "}\n"
        #table_latex_string += "\\\\ \\hline \n"
        
    print(table_latex_string)
    
    plt.xlabel("FAR", fontsize=15)
    plt.ylabel("FRR", fontsize=15)
    plt.xlim(0, 0.2)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.savefig(f"{file}.png")
    plt.show()