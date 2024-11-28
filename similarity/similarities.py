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
    

def convert_data_to_classes(user_list, data) -> dict[int, User]:
    user_objects = dict()
    for user_id in user_list:
        user_objects[int(user_id)] = User(int(user_id))
    for x in data:
        user_objects[int(x[1])].set_rating(int(x[2]),int(x[3]))
    return user_objects

def calc_similaries_vector_for_user(main_user_id, user_data: dict[int,User]):
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
            for missing_movie_id in [x for x in user_data[main_user_id].movie_ratings if user_data[main_user_id].movie_ratings[x] is None]:
                if user_data[user_id].has_rating(missing_movie_id):
                    if(missing_movie_id == 1): print(f'{user_id}:{missing_movie_id}')
                    similarities_dict[missing_movie_id].append([similarity_sum_distance / len(same_movies_id)])
                    labels_dict[missing_movie_id].append(user_data[user_id].movie_ratings[missing_movie_id])
    return similarities_dict, labels_dict


def kNN_similarities_predict_single(k, train_dict, movie_id, known_classes, distance):
    return known_classes.movie_ratings[movie_id] == kNN_predict(k, train_dict, movie_id, distance)

def kNN_train_and_predict_all(user_data, k, split_threshold):
    train_model_data = deepcopy(user_data)
    for user_id in user_data:
        train_model_data[user_id].reset_ratings(split_threshold)
    all_users_accuracy = []
    for user_id in user_data:
        similarities_dict, labels_dict = calc_similaries_vector_for_user(user_id, train_model_data)
        user_true = []
        for movie_id in similarities_dict:
            if(len(similarities_dict[movie_id]) > 0):
                model = kNN_train(similarities_dict[movie_id], labels_dict[movie_id])
                user_true.append(int(kNN_similarities_predict_single(k, model, movie_id, user_data[user_id], similarity_distance)))
                all_users_accuracy.append(user_true[-1])
        print(f"{user_id}: {sum(user_true)/len(user_true)}")
    print(sum(all_users_accuracy)/len(all_users_accuracy))
            


def similarity_distance(train,predict):
    return train[0]


def train(split_threshold = 0.2, k = 8):
    user_list = get_user_list()
    train_data = get_train_data()
    train_data_classes = convert_data_to_classes(user_list,train_data)
    kNN_train_and_predict_all(train_data_classes, k, split_threshold)

def classify():
    pass

if __name__ == '__main__':
    train()