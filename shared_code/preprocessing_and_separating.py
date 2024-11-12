from json import load
from csv import writer, reader
from collections import defaultdict
from datetime import datetime
from helpers import save_movie_json
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np

FEATURE_KEYWORDS = ['budget', 'original_language','popularity', 'release_date', 'runtime','production_companies', 'genres','spoken_languages_count','keywords','popular_actors']
PURGEABLE_KEYWORDS = ['production_companies','genres','keywords','popular_actors']
BOOL_KEYWORDS = ['adult']
BOOL_LIST_KEYWORDS = ['original_language']

def purge_nonrelevant_feature_values(movies_data, key, min_no_of_occurences = 5, max_no_of_occurences = 55):
    count_dict = defaultdict(lambda: 0)
    for movie in movies_data:
        values = str(movie[key]).split(',')
        for value in values:
            count_dict[value] += 1
    for movie in movies_data:
        values = str(movie[key]).split(',')
        movie[key] = ",".join(v for v in str(movie[key]).split(',') if count_dict[v] >= min_no_of_occurences and count_dict[v] <= max_no_of_occurences)
    with open(f'feature_lists/purged_{key}.txt','w') as key_f:
        key_f.write(",".join([v for v in count_dict if count_dict[v] >= min_no_of_occurences and count_dict[v] <= max_no_of_occurences]))
    return movies_data

def create_list_of_unique_values_for_key(movies_data,key):
    with open(f'feature_lists/purged_{key}.txt','w') as key_f:
        key_f.write(",".join(list({m[key] for m in movies_data})))

def create_movie_feature_vectors(movies_data):
    vector_list = []
    feature_list = []
    for movie in movies_data:
        vector = []
        for feature in FEATURE_KEYWORDS:
            if feature in PURGEABLE_KEYWORDS or feature in BOOL_LIST_KEYWORDS:
                movie_list_feature_list = str(movie[feature]).split(',') if feature in PURGEABLE_KEYWORDS else [str(movie[feature])]
                with open(f'feature_lists/purged_{feature}.txt') as feature_f:
                    list_features = feature_f.read().split(',')
                    for list_feature in list_features:
                        vector.append(int(list_feature in movie_list_feature_list))
                        if len(vector_list) == 0:
                            feature_list.append(f'{feature} - {list_feature}')
            else:
                if feature == 'release_date':
                    split_date = str(movie[feature]).split('-')
                    vector.append(int(split_date[0])*365+int(split_date[1])*30+int(split_date[2]))
                else:
                    vector.append(movie[feature])
                if len(vector_list) == 0:
                    feature_list.append(feature)
        vector_list.append(vector)
    return vector_list,feature_list

def save_feature_vectors_to_csv(filename, vector_list, feature_list):
    with open(filename,'w',newline='') as csv_f:
        write = writer(csv_f)
        write.writerow(feature_list)
        write.writerows(vector_list)

def generate_features_and_ratings_separate_by_user(vector_list):
    user_datasets = defaultdict(lambda: [])
    Path("separated_feature_vectors").mkdir(exist_ok=True)
    with open("task_data/train.csv") as task_f:
        read = reader(task_f,delimiter=';')
        for rate in read:
            user_datasets[int(rate[1])].append(np.append(vector_list[int(rate[2])-1],[int(rate[3]),int(rate[2])]))
    for user_id in user_datasets:
        with open(f"separated_feature_vectors/{user_id}.csv",'w',newline='') as csv_f:
            write = writer(csv_f)
            write.writerows(user_datasets[user_id])

def generate_separate_ratings_by_user():
    user_datasets = defaultdict(lambda: [])
    Path("separated_ratings").mkdir(exist_ok=True)
    with open("task_data/train.csv") as task_f:
        read = reader(task_f,delimiter=';')
        for rate in read:
            user_datasets[int(rate[1])].append(rate)
    for user_id in user_datasets:
        with open(f"separated_ratings/{user_id}.csv",'w',newline='') as csv_f:
            write = writer(csv_f,delimiter=';')
            write.writerows(user_datasets[user_id])

def normalize_dataset(vector_list):
    return StandardScaler().fit_transform(vector_list)

if __name__ == '__main__':
    generate_separate_ratings_by_user()
    with open('movies_data.json') as movies_f:
        Path("feature_lists").mkdir(exist_ok=True)
        Path("movie_data").mkdir(exist_ok=True)
        movies_data = load(movies_f)
        for keyword in PURGEABLE_KEYWORDS:
            movies_data = purge_nonrelevant_feature_values(movies_data, keyword)
        for keyword in BOOL_LIST_KEYWORDS:
            create_list_of_unique_values_for_key(movies_data, keyword)
        save_movie_json(movies_data,'movie_data/purged_movies_data.json')
        vector_list, feature_list =create_movie_feature_vectors(movies_data)
        save_feature_vectors_to_csv('movie_data/unnormalized_movie_feature_vector.csv', vector_list, feature_list)
        save_feature_vectors_to_csv('movie_data/normalized_movie_feature_vector.csv', normalize_dataset(vector_list), feature_list)
        generate_features_and_ratings_separate_by_user(normalize_dataset(vector_list))