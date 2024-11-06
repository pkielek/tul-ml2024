import random
import numpy as np


# def fake_data(num_obj, num_feat):
#     if num_obj == 1:
#         obj =  np.empty(num_feat)
#         for i in range(num_feat):
#             obj[i] = random.uniform(0, 1)
#         return obj, random.randint(0,5)
#     features = np.empty((num_obj,num_feat))
#     classes = np.empty(num_obj)
#     for i in range(num_obj):
#         for j in range(num_feat):
#             features[i][j] = random.uniform(0, 1)
#         classes[i] = random.randint(0,5)
#     return features, classes
        

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
    accuracy = predicted / len(to_predict)



# data, classes = fake_data(20, 10)
# model = kNN_train(data, classes)
# to_predict, known_class = fake_data(1, 10)
# predicted = kNN_predict(5, model, to_predict, manhattan_distance)
# print(predicted)
# print(known_class)