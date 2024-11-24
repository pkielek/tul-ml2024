
import numpy as np

def one_class_saturation(data, label):
    count = 0
    for entry in data:
        count +=1 if entry[-2] == label else 0
    return count / len(data)

def all_classes_saturation(data):
    _, class_counts = np.unique(np.array(data).T[-2], return_counts=True)
    return class_counts / len(data)

# determine if any class % in data is above threshold
def class_over_threshold(data, threshold):
    labels = np.unique(np.array(data).T[-2])
    for label in labels:
        if one_class_saturation(data, label) >= threshold: return label
    return None

def entropy(data):
    result = 0
    probabilities = all_classes_saturation(data)
    for probability in probabilities:
        result += probability * np.log(probability)
    return result * -1

def calculate_split_metric(data_left, data_right, method):
    total = len(data_left) + len(data_right)
    return (((len(data_left) / total) * method(data_left)) + ((len(data_right) / total) * method(data_right))) * -1

def gini_impurity(y):
    
    probabilities = all_classes_saturation(y)
    
    # Calculate Gini Impurity
    gini = 1 - np.sum(probabilities ** 2)
    return gini

def calculate_info_gain(data, data_left, data_right, method):
    return method(data) - calculate_split_metric(data_left, data_right, method)