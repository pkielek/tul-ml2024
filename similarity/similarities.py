from time import sleep
import numpy as np
from shared_code.helpers import get_user_list,get_train_data, get_task_data
from knn.kNN import kNN_predict, kNN_train
from random import random
from copy import deepcopy


class User:

    def __init__(self, user_id):
        self.movie_ratings = dict()
        self.user_id = user_id

    def set_rating(self,movie_id, rating):
        self.movie_ratings[movie_id] = rating if rating != -1 else None

    def has_rating(self,movie_id):
        return movie_id in self.movie_ratings and self.movie_ratings[movie_id] is not None
    
    def get_keys_with_ratings(self):
        return {k for k in self.movie_ratings if self.movie_ratings[k] is not None}

    def reset_ratings(self, probability):
        for movie_rating in self.movie_ratings:
            threshold = random()
            if threshold <= probability:
                self.movie_ratings[movie_rating] = None
    
    def __repr__(self):
        return f'{self.user_id}: {self.movie_ratings}'
    
    def __str__(self):
        return f'{self.user_id}: {self.movie_ratings}'
    
def similarity_distance(train,predict):
    return train[0]

def convert_data_to_classes(user_list, data) -> dict[int, User]:
    user_objects = dict()
    for user_id in user_list:
        user_objects[int(user_id)] = User(int(user_id))
    for x in data:
        user_objects[int(x[1])].set_rating(int(x[2]),int(x[3]))
    return user_objects

def calc_similarities_vector_for_user(main_user_id, user_data: dict[int,User], classify = False):
    similarities_dict = dict()
    labels_dict = dict()
    for i in range(1,201):
        similarities_dict[i] = []
        labels_dict[i] = []
    for user_id in user_data:
        if user_id != main_user_id:
            same_movies_id = user_data[user_id].get_keys_with_ratings() & user_data[main_user_id].get_keys_with_ratings()
            similarity_sum_distance = 0
            for movie_id in same_movies_id:
                similarity_sum_distance += int(user_data[user_id].movie_ratings[movie_id] != user_data[main_user_id].movie_ratings[movie_id])
            for missing_movie_id in range(1,201) if classify else [x for x in user_data[main_user_id].movie_ratings if user_data[main_user_id].movie_ratings[x] is None]:
                if user_data[user_id].has_rating(missing_movie_id):
                    similarities_dict[missing_movie_id].append([similarity_sum_distance / len(same_movies_id)])
                    labels_dict[missing_movie_id].append(user_data[user_id].movie_ratings[missing_movie_id])
    return similarities_dict, labels_dict


def kNN_similarities_predict_single(k, train_dict, movie_id, known_classes, distance):
    return known_classes.movie_ratings[movie_id] == kNN_predict(k, train_dict, movie_id, distance)

def kNN_train_and_validate_all(user_data, k, split_threshold, print_all = True):
    train_model_data = deepcopy(user_data)
    for user_id in user_data:
        train_model_data[user_id].reset_ratings(split_threshold)
    all_users_accuracy = []
    for user_id in user_data:
        similarities_dict, labels_dict = calc_similarities_vector_for_user(user_id, train_model_data)
        user_true = []
        for movie_id in similarities_dict:
            if(len(similarities_dict[movie_id]) > 0):
                model = kNN_train(similarities_dict[movie_id], labels_dict[movie_id])
                user_true.append(int(kNN_similarities_predict_single(k, model, movie_id, user_data[user_id], similarity_distance)))
                all_users_accuracy.append(user_true[-1])
        if print_all:
            print(f"{user_id} accuracy: {sum(user_true)/len(user_true)}")
    total_accuracy = sum(all_users_accuracy)/len(all_users_accuracy)
    print(f"Total accuracy for k={k} : {total_accuracy}")
    return total_accuracy

def kNN_classify_all(train_data, k):
    user_similarities_dicts = dict()
    user_labels_dicts = dict()
    print("Generating similiarities")
    for user_id in train_data:
        print("\t" + str(user_id))
        user_similarities_dicts[user_id], user_labels_dicts[user_id] = calc_similarities_vector_for_user(user_id, train_data, classify=True)
    task_data = get_task_data()
    task_results = np.empty((0, 4), dtype=int)
    print("Classifying")
    for row in task_data:
        print("\t" + str(row[0]))
        user_id, movie_id = row[1], row[2]
        predicted = int(kNN_predict(k, kNN_train(user_similarities_dicts[user_id][movie_id], user_labels_dicts[user_id][movie_id]), movie_id, similarity_distance))        
        task_results = np.append(task_results, [np.array([row[0], row[1], row[2], predicted], dtype=int)], axis=0)

    np.savetxt('similarity/task_results.csv', task_results, delimiter=';', fmt="%d")

k = 8
split_threshold = 0.2
if __name__ == '__main__':
    user_list = get_user_list()
    train_data = get_train_data()
    train_data_classes = convert_data_to_classes(user_list,train_data)
    kNN_train_and_validate_all(train_data_classes, k, split_threshold)
    kNN_classify_all(train_data_classes, k)
