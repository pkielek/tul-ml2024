from csv import writer
from json import dumps
from pathlib import Path

import numpy as np


def save_movie_json(movie_data, filename):
    movies_json = dumps(movie_data,indent=3)
    with open(filename,'w') as out:
        out.write(movies_json)

def save_user_features_csv(dirname, user_id, data):
    Path(dirname).mkdir(exist_ok=True,parents=True)
    with open(f"{dirname}/{user_id}.csv", 'w', newline='') as csv_f:
        write = writer(csv_f)
        write.writerows(data)

def load_user_feature_vector_from_file(user_id, directory = 'separated_feature_vectors'):
    data = np.genfromtxt(f'{directory}/{user_id}.csv', delimiter=',', dtype=float)
    if 'vectors' not in directory: # artbitraly difference for directories where labels and movie_ids are included
        return data
    return data[:, :-2], data[:, -2], data[:, -1] # x, y and movie_ids

def get_user_list():
    return np.unique(np.genfromtxt('task_data/train.csv', delimiter=';', dtype=int)[:,1])

def get_task_data():
    return np.genfromtxt('task_data/task.csv', delimiter=';', dtype=int)

def load_movie_feature_vectors(unnormalized = False):
    data = np.genfromtxt(f'movie_data/{"un" if unnormalized else ""}normalized_movie_feature_vector.csv', delimiter=',', dtype=float)
    return data[1:]

def load_movie_feature_vector_names(unnormalized = False):
    data = np.genfromtxt(f'movie_data/{"un" if unnormalized else ""}normalized_movie_feature_vector.csv', delimiter=',', dtype=str)
    return data[0,:]