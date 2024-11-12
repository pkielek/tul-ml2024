from csv import reader
from json import dumps

import numpy as np


def save_movie_json(movie_data, filename):
    movies_json = dumps(movie_data,indent=3)
    with open(filename,'w') as out:
        out.write(movies_json)

def load_user_feature_vector_from_file(user_id):
    data = np.genfromtxt(f'separated_feature_vectors/{user_id}.csv', delimiter=',', dtype=float)
    return data[:, :-2], data[:, -2], data[:, -1] # x, y and movie_ids

def get_user_list():
    return np.unique(np.genfromtxt('task_data/train.csv', delimiter=';', dtype=int)[:,1])