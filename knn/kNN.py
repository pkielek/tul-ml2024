import numpy as np
from sklearn.model_selection import train_test_split
from shared_code.filters import correlation_filter, mutual_information_filter, greedy_backwards_feature_selection, greedy_forward_feature_selection
from shared_code.helpers import load_user_feature_vector_from_file, get_user_list, save_user_features_csv

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

def single_user_test(user_id, test_size, method, k, filter, greedy_backwards_fs = False, greedy_forwards_fs = False):
    movie_features, known_classes, _ = load_user_feature_vector_from_file(user_id)
    if filter is not None:
        movie_features = filter(movie_features, known_classes)
    accuracy = kNN_train_and_predict_all(movie_features, known_classes, k, method, test_size)
    if greedy_backwards_fs and greedy_forwards_fs:
        raise ValueError("Can't use both forward and backwards feature selection")
    if greedy_backwards_fs or greedy_forwards_fs:
        func = greedy_backwards_feature_selection if greedy_backwards_fs else greedy_forward_feature_selection
        print(f"user:{user_id}")
        accuracy, selected_features = func(accuracy if greedy_backwards_fs else 0.0, 
                                            movie_features, 
                                            np.arange(movie_features.shape[1]) if greedy_backwards_fs else np.empty([],dtype=int), 
                                            lambda movie_features: kNN_train_and_predict_all(movie_features, known_classes, k, method, test_size))
        save_user_features_csv(f'kNN/greedy_{"backwards" if greedy_backwards_fs else "forwards"}_features',user_id, movie_features[:, selected_features])
        save_user_features_csv(f'kNN/greedy_{"backwards" if greedy_backwards_fs else "forwards"}_features_indices',user_id, [[f] for f in selected_features])
    return accuracy

def main(): # for testing you can add additional for loops for test_size, method, k-value or filter method, to verify if current chosen parameters are the best
    users = get_user_list()
    accuracy_list = []
    for user_id in users:
        accuracy = single_user_test(user_id, 0.25, manhattan_distance, 8, correlation_filter, True)
        accuracy_list.append(accuracy)
    print(f"Total accuracy = {np.average(accuracy_list)}")


if __name__ == '__main__':
    main()