
import numpy as np


def gini_impurity(y):
    _, class_counts = np.unique(y, return_counts=True)
    probabilities = class_counts / len(y)
    
    # Calculate Gini Impurity
    gini = 1 - np.sum(probabilities ** 2)
    return gini