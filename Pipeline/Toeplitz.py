from scipy.linalg import toeplitz
from numpy import linalg as LA
import numpy as np


def toeplitz_coefficients(sequences):
    """
    Takes arbitrary number of sequences of equal lengths
    Then iterates over positions in those sequences and counts euclidean norm for elements on the same positions
    Then from vector created in such way, toeplitz coefficients matrix is created
    """

    if not all([len(seq) == len(sequences[0]) for seq in sequences]):
        raise Exception("Sequnces not of equal length")
        
    vect_to_toeplitz = np.zeros(len(sequences[0]))
    
    for seq in sequences:
        vect_to_toeplitz += np.power(seq, 2)
        
    vect_to_toeplitz = np.sqrt(vect_to_toeplitz)
    
    mat = toeplitz(vect_to_toeplitz)
    n = mat.shape[0]

    ret = []
    
    for k in range(n):
        submatrix = mat[:k+1, :k+1]
        w, _ = LA.eig(submatrix)
        ret.append(min(w))
        
    return ret
