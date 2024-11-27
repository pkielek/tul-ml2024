from trees.trees_v2 import Tree, create_tree, get_conditions_from_file
from trees.evaluation import gini_impurity
import numpy as np
from shared_code.helpers import get_user_list, load_user_feature_vector_from_file, get_task_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

GLOBAL_THRESHOLD = 0.95

def forest_classify(forest_ensemble, entry):
    results = np.zeros(6)
    for tree in forest_ensemble:
        results[tree.classify(entry)]+=1
    return np.argmax(results)

def forest_classify_test(forest_ensemble, data):
    count = 0
    for entry in data:
        if forest_classify(forest_ensemble, entry) == entry[-2]: count += 1
    return count / len(data)

def generate_forest(data, conditions, n_trees, num_features, method):
    ensemble_trees = []
    for _ in range(n_trees):
        random_train = data[np.random.choice(len(data), len(data)),:]
        random_conditions = list(np.random.choice(len(conditions)-2, num_features, False))
        ensemble_trees.append(create_tree(random_train, conditions, GLOBAL_THRESHOLD, method, random_conditions))

    return ensemble_trees
def forest_training(n_trees = 150, num_features = 42, method = gini_impurity):
    global_acc = 0

    user_list = get_user_list()
    conditions = get_conditions_from_file()
    for user_id in user_list:
        user_movies = load_user_feature_vector_from_file(user_id, 'trees/separated_binned_feature_vectors', load_all = True)
        train, test = train_test_split(user_movies, test_size=0.15, random_state=42)
        ensemble_trees = generate_forest(train, conditions, n_trees, num_features, method)
        acc = forest_classify_test(ensemble_trees, test)
        
        global_acc += acc
        print(str(user_id) + " = " + str(acc))
    
    print("Global accuracy = " + str(global_acc / len(user_list)))

def forest_train_and_predict(n_trees = 150, num_features = 35, method = gini_impurity):
    forest_dict = {}

    movie_data = np.genfromtxt("trees/binned_movie_feature_vector.csv", delimiter=',', dtype=int)
    user_list = get_user_list()
    conditions = get_conditions_from_file()

    print("Creating forests")
    for user_id in user_list:
        print("\t" +str(user_id))
        user_movies = load_user_feature_vector_from_file(user_id, 'trees/separated_binned_feature_vectors', load_all = True)
        forest_dict[int(user_id)] = generate_forest(user_movies, conditions, n_trees, num_features, method)
    
    task_data = get_task_data()
    task_results = np.empty((0, 4), dtype=int)

    print("Classifying")
    for row in task_data:
        print("\t" + str(row[0]))
        user_id, movie_id = row[1], row[2]
        predicted = forest_classify(forest_dict[int(user_id)],movie_data[int(movie_id)])
        task_results = np.append(task_results, [np.array([row[0], row[1], row[2], predicted], dtype=int)], axis=0)

    np.savetxt('trees/forest_task_results_.csv', task_results, delimiter=';', fmt="%d")


def analyze_results():
    results = np.genfromtxt('trees/forest_task_results_.csv', delimiter=';', dtype=int)
    count = 0
    count_labels = np.zeros(6)
    percentages = []
    current_user_id = -1
    for row in results:
        user_id, label = row[1], row[3]
        if current_user_id != user_id:
            percentages.append(np.amax(count_labels) / count)
            count_labels = np.zeros(6)
            count = 0
            current_user_id = user_id
        count_labels[label] += 1
        count += 1
    print(percentages)
    plt.hist(percentages)
    plt.show()

if __name__ == '__main__':
    # forest_training()
    forest_train_and_predict()
    analyze_results()