
import numpy as np
import os
from sklearn.model_selection import train_test_split
from copy import deepcopy
from trees.evaluation import *

from shared_code.helpers import get_task_data

class Tree:

    condition = -1
    index = -1
    left = None
    right = None

    def set_condition(self, condition): self.condition = condition
    def set_index(self, index): self.index = index
    def set_left(self, left): self.left = left
    def set_right(self, right): self.right = right
    
    # test elements against condition and split according to result
    def split_data(self, data):

        true_list = []
        false_list = []

        for entry in data:

            if entry[self.index]:
                true_list.append(entry)
            else:
                false_list.append(entry)

        return true_list, false_list
    
    def classify(self, entry):
        if type(self.condition) == int:
            return self.condition
        
        result = entry[self.index]
        if result:
            return self.left.classify(entry)
        
        return self.right.classify(entry)

# return list of conditions imported from file, in form of lambda expressions
def get_conditions_from_file(path):
    return np.genfromtxt(path, delimiter=',', dtype=str)[0]

def create_tree(data, conditions, threshold, method):

    majority_class = class_over_threshold(data, threshold)
    if majority_class is not None:
        tree = Tree()
        tree.set_condition(int(majority_class))
        return tree

    tree = Tree()
    info_gain = []
    used_conditions = []

    for i, condition in enumerate(conditions):
        if i in used_conditions:
            info_gain.append(0)
            continue
        tree.set_condition(condition)
        tree.set_index(i)
        left, right = tree.split_data(data)
        if len(left) == 0 or len(right) == 0:
            info_gain.append(0)
        else:
            info_gain.append(calculate_info_gain(left, right, data, method))
    
    index = info_gain.index(max(info_gain))
    best_condition = conditions[index]
    used_conditions.append(index)
    tree.set_condition(best_condition)
    tree.set_index(index)
    left, right = tree.split_data(data)

    tree.set_left(create_tree(left, conditions, threshold, method))
    tree.set_right(create_tree(right, conditions, threshold, method))

    return tree

def classify_test(tree, data):
    count = 0
    for entry in data:
        guess = tree.classify(entry)
        if guess == entry[-2]: count+=1
    return count / len(data)

def find_best_thresholds(method):

    thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    all_path = "trees/separated_binned_feature_vectors"
    global_acc = 0
    file_count = 0

    save_thresholds = []

    for filename in os.listdir(all_path):

        path = os.path.join(all_path, filename)
        conditions = get_conditions_from_file("trees/binned_movie_feature_vector.csv")
        user_movies = np.genfromtxt(path, dtype=int, delimiter=",")
        train, test = train_test_split(user_movies, test_size=0.1, random_state=42)

        best_acc = 0
        best_threshold = 0
        
        for threshold in thresholds:
            tree = create_tree(train, conditions, threshold, method)
            acc = classify_test(tree, test)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        save_thresholds.append([filename[:-4], str(best_threshold)])
        global_acc += best_acc
        file_count += 1
        print(filename + " = " + str(best_acc) + "; threshold = " + str(best_threshold))
    
    print("Global accuracy = " + str(global_acc / file_count))
    np.savetxt(f"trees/best_thresholds_{method.__name__}.csv", save_thresholds, delimiter=',', fmt="%s")

def train_and_predict_all(method, custom_thresholds = True):

    all_path = "trees/separated_binned_feature_vectors"

    tree_dict = {}
    thresh_dict = {}

    movie_data = np.genfromtxt("trees/binned_movie_feature_vector.csv", delimiter=',', dtype=int)

    if custom_thresholds:
        thresholds = np.genfromtxt(f"trees/best_thresholds_{method.__name__}.csv", delimiter=',', dtype=float)
        for threshold in thresholds:
            thresh_dict[threshold[0]] = threshold[1]

    print("Creating trees")
    for filename in os.listdir(all_path):

        print("\t" + filename)

        path = os.path.join(all_path, filename)
        conditions = get_conditions_from_file("trees/binned_movie_feature_vector.csv")
        user_movies = np.genfromtxt(path, dtype=int, delimiter=",")
        threshold = thresh_dict[int(filename[:-4])] if custom_thresholds else 1.0
        tree = create_tree(user_movies, conditions, threshold, method)
        tree_dict[int(filename[:-4])] = tree
    
    task_data = get_task_data()
    task_results = np.empty((0, 4), dtype=int)

    print("Classifying")
    for row in task_data:

        print("\t" + str(row[0]))

        user_id, movie_id = row[1], row[2]
        predicted = tree_dict[int(user_id)] if type(tree_dict[int(user_id)]) == int else tree_dict[int(user_id)].classify(movie_data[int(movie_id)])
        task_results = np.append(task_results, [np.array([row[0], row[1], row[2], predicted], dtype=int)], axis=0)

    name = f'trees/task_results_{"custom_threshold" if custom_thresholds else "same_threshold"}_{method.__name__}.csv' 

    np.savetxt(name, task_results, delimiter=';', fmt="%d")


if __name__ == '__main__':

    # find_best_thresholds(gini_impurity)
    train_and_predict_all(gini_impurity, False)