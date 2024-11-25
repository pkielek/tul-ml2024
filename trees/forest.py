from trees.trees_v2 import Tree, create_tree, get_conditions_from_file
from trees.evaluation import gini_impurity
import numpy as np
from shared_code.helpers import get_user_list, load_user_feature_vector_from_file
from sklearn.model_selection import train_test_split

GLOBAL_THRESHOLD = 0.95


def forest_classify_test(forest_ensemble, data):
    results = np.zeros(6)
    count = 0
    for entry in data:
        results = np.zeros(6)
        for tree in forest_ensemble:
            results[tree.classify(entry)]+=1
        guess = np.argmax(results)
        if guess == entry[-2]: count+=1
    return count / len(data)

def forest_training(n_trees = 1500, num_features = 13, method = gini_impurity):
    global_acc = 0

    user_list = get_user_list()
    conditions = get_conditions_from_file()
    for user_id in user_list:
        user_movies = load_user_feature_vector_from_file(user_id, 'trees/separated_binned_feature_vectors', load_all = True)
        train, test = train_test_split(user_movies, test_size=0.15, random_state=42)
        ensemble_trees = []
        for _ in range(n_trees):
            random_train = train[np.random.choice(len(train), len(train)),:]
            random_conditions = list(np.random.choice(len(conditions)-2, num_features, False))
            ensemble_trees.append(create_tree(random_train, conditions, GLOBAL_THRESHOLD, method, random_conditions))

        acc = forest_classify_test(ensemble_trees, test)
        
        global_acc += acc
        print(str(user_id) + " = " + str(acc))
    
    print("Global accuracy = " + str(global_acc / len(user_list)))

if __name__ == '__main__':
    forest_training()