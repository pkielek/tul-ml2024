from json import dumps
from pathlib import Path
from requests import get
from csv import reader
from os import getenv
from dotenv import load_dotenv

from helpers import save_movie_json

load_dotenv()

MOVIE_FILE = 'task_data/movie.csv'

details_url = lambda id : f"https://api.themoviedb.org/3/movie/{id}"
keywords_url = lambda id : details_url(id) + "/keywords"
credits_url = lambda id : details_url(id) + "/credits"

def tmdb_api_response(id, url):
    headers = {"accept": "application/json",
           "Authorization": f"Bearer {getenv('API_READ_KEY')}"}
    return get(url(id), headers = headers)

def fetch_movie_data(init_movie_data):
    copy_keywords = ['budget', 'original_language','popularity', 'release_date', 'runtime']
    list_keywords = ['production_companies', 'genres']

    movie_json = {'id': init_movie_data[0], 'movie_id': init_movie_data[1], 'title': init_movie_data[2]}
    movie_id = init_movie_data[1]
    print(f'Processing {init_movie_data[0]} movie ({init_movie_data[2]})')
    details = tmdb_api_response(movie_id, details_url).json()
    for key in copy_keywords:
        movie_json[key] = details[key]
    for key in list_keywords:
        movie_json[key] = ",".join([x['name'] for x in details[key]])
    movie_json['spoken_languages_count'] = len(details['spoken_languages'])
    keywords = tmdb_api_response(movie_id, keywords_url).json()
    movie_json['keywords'] = ",".join([kw['name'] for kw in keywords['keywords']])
    credits = tmdb_api_response(movie_id, credits_url).json()
    movie_json['popular_actors'] = ",".join([x['name'] for x in sorted(credits['cast'], key = lambda x : x['popularity'], reverse= True)[:10]])
    return movie_json

def fetch_data_for_movies_from_file(filename):
    with open(filename) as movie_f:
        movies = reader(movie_f,delimiter=';')
        return [fetch_movie_data(movie) for movie in movies]

if __name__ == '__main__':
    movie_data = fetch_data_for_movies_from_file(MOVIE_FILE)
    Path("movie_data").mkdir(exist_ok=True)
    save_movie_json(movie_data, 'movie_data/movies_data.json')

