from csv import reader
from json import dumps

import numpy as np


def save_movie_json(movie_data, filename):
    movies_json = dumps(movie_data,indent=3)
    with open(filename,'w') as out:
        out.write(movies_json)

def load_user_feature_vector_from_file(file_path):
    x = []
    y = []
    movie_ids = []
    with open(file_path) as f:
        read = reader(f)
        for rating in read:
            movie_ids.append(int(float(rating[-1])))
            y.append(int(float(rating[-2])))
            x.append(rating[:-2])
    return np.array(x,dtype='float64'), np.array(y), movie_ids