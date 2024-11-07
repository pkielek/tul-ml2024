import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split        

def get_user_list():
    train_data = genfromtxt('task_data/train.csv', delimiter=';', dtype=int)
    temp = np.unique(np.transpose(train_data)[1])
    return temp

def get_scores_per_user(user_id):
    ret = np.empty((0,4), dtype=int)
    train_data = genfromtxt('task_data/train.csv', delimiter=';', dtype=int)
    for row in train_data:
        if row[1] == user_id:
            ret = np.append(ret, [row], axis=0)
    return ret

def get_movies_features(movie_ids):
    movie_data = genfromtxt('movie_data/normalized_movie_feature_vector.csv', delimiter=',', dtype=float)
    ret = np.empty((0,len(movie_data[0])), dtype=float)
    # print(movie_data)
    for i, row in enumerate(movie_data):
        if i == 0: continue
        if i in movie_ids:
            ret = np.append(ret, [row], axis=0)
    return ret

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
    ret = dict()
    for train, actual_class in zip(train_set, classes):
        ret[train.tobytes()] = actual_class
    return ret

def kNN_predict(k, train_dict, to_predict, distance):
    distances = dict()
    class_pred = dict.fromkeys([0,1,2,3,4,5], 0)
    for key, val in train_dict.items():
        obj = np.frombuffer(key, float)
        distances[(key, val)] = distance(obj, to_predict)
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
    
k_list = range(5, 15)
test_sizes = [0.1, 0.2, 0.25, 0.3, 0.33, 0.4]
distance_methods = [manhattan_distance]

def main():
    users = get_user_list()
    print(users)
    for user_id in users:
        scores = get_scores_per_user(user_id)
        movie_features = get_movies_features(np.transpose(scores)[2])
        # tutaj można dodać wybór cech
        known_classes = np.transpose(scores)[3]
        for test_size in test_sizes:
            print(f'test_size = {test_size}')
            X_train, X_test, y_train, y_test = train_test_split(movie_features, known_classes, test_size=0.20, random_state=42)
            model = kNN_train(X_train, y_train)
            for method in distance_methods:
                for k in k_list:
                    accuracy = kNN_predict_all(k, model, X_test, y_test, method)
                    print(f'distance = {method.__name__}, k = {k}, accuracy = {accuracy}')

def single_user_test():
    users = get_user_list()
    scores = get_scores_per_user(users[4])
    movie_features = get_movies_features(np.transpose(scores)[2])
    # tutaj można dodać wybór cech
    known_classes = np.transpose(scores)[3]

    X_train, X_test, y_train, y_test = train_test_split(movie_features, known_classes, test_size=0.20, random_state=42)

    model = kNN_train(X_train, y_train)
    accuracy = kNN_predict_all(5, model, X_test, y_test, manhattan_distance)
    print(accuracy)


if __name__ == '__main__':
    single_user_test()
    # main()