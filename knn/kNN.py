import numpy as np
from sklearn.model_selection import train_test_split
from shared_code.filters import correlation_filter, greedy_backwards_feature_selection
from shared_code.helpers import load_movie_feature_vectors, load_user_feature_vector_from_file, get_user_list, save_user_features_csv, get_task_data

def manhattan_distance(trained, to_predict):
    if (len(trained) != len(to_predict)):
        return None
    all = 0
    for first, second in zip(trained, to_predict):
        all += abs(first - second)
    return all

def kNN_train(train_set, classes):
    if (len(train_set) != len(classes)):
        return None
    return [(train, actual_class) for train, actual_class in zip(train_set,classes)]

def kNN_predict(k, train_dict, to_predict, distance):
    class_pred = dict.fromkeys([0,1,2,3,4,5], 0)
    distances = {(i, actual_class): distance(train, to_predict) for i, (train, actual_class) in enumerate(train_dict)}
    k_closest = sorted(distances, key=distances.get, reverse=False)[:k]
    for tupple in k_closest:
        class_pred[int(tupple[1])] += 1
    predicted = sorted(class_pred, key=class_pred.get, reverse=True)[:1][0]
    return predicted

def kNN_predict_all(k, train_dict, to_predict, known_classes, distance):
    predicted = 0
    for elem, known_class in zip(to_predict, known_classes):
        pred = kNN_predict(k, train_dict, elem, distance)
        if known_class == pred: predicted += 1
    return predicted / len(to_predict)

def kNN_train_and_predict_all(movie_features, known_classes, k, method, test_size):
    X_train, X_test, y_train, y_test = train_test_split(movie_features, known_classes, test_size=test_size, random_state=42)
    model = kNN_train(X_train, y_train)
    return kNN_predict_all(k, model, X_test, y_test, method)

def kNN_task_predict_all():
    task_data = get_task_data()
    directory = 'knn/greedy_backwards_features'
    current_user_id = None
    movies_features = load_movie_feature_vectors()
    task_results = np.empty((0, 4), dtype=int)
    for row in task_data:
        user_id, movie_id = row[1], row[2]
        if user_id != current_user_id:
            current_user_id = user_id
            _, y_train, _ = load_user_feature_vector_from_file(user_id)
            x_train = load_user_feature_vector_from_file(user_id, directory)
            model = kNN_train(x_train, y_train)
            selected_features_indices = np.genfromtxt(f'{directory}_indices/{user_id}.csv', delimiter=',', dtype=int)
        movie_features = movies_features[movie_id-1, selected_features_indices]
        predicted = kNN_predict(8, model, movie_features, manhattan_distance)
        task_results = np.append(task_results, [np.array([row[0], row[1], row[2], predicted], dtype=int)], axis=0)
    np.savetxt('knn/task_results.csv', task_results, delimiter=';', fmt="%d")

def single_user_test(user_id, test_size, method, k, filter, greedy_backwards_fs = False):
    movie_features, known_classes, _ = load_user_feature_vector_from_file(user_id)
    selected_indices = filter(movie_features, known_classes) if filter is not None else np.arange(movie_features.shape[1])
    if len(selected_indices) == 0:
        selected_indices = np.array([0,1])
    accuracy = kNN_train_and_predict_all(movie_features[:, selected_indices], known_classes, k, method, test_size)
    if greedy_backwards_fs:
        print(f"user:{user_id}")
        accuracy, selected_features = greedy_backwards_feature_selection(accuracy, 
                                            movie_features, 
                                            selected_indices, 
                                            lambda movie_features: kNN_train_and_predict_all(movie_features, known_classes, k, method, test_size))
        save_user_features_csv(f'kNN/greedy_backwards_features',user_id, movie_features[:, selected_features])
        save_user_features_csv(f'kNN/greedy_backwards_features_indices',user_id, [[f] for f in selected_features])
    return accuracy

def train(): # for testing you can add additional for loops for test_size, method, k-value or filter method, to verify if current chosen parameters are the best
    users = get_user_list()
    accuracy_list = []
    for user_id in users:
        accuracy = single_user_test(user_id, 0.25, manhattan_distance, 8, correlation_filter, True)
        accuracy_list.append(accuracy)
    print(f"Total accuracy = {np.average(accuracy_list)}")

def test():
    kNN_task_predict_all()

if __name__ == '__main__':
    # train()
    test()