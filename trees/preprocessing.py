from shared_code.helpers import load_movie_feature_vectors, load_movie_feature_vector_names
from shared_code.preprocessing_and_separating import generate_features_and_ratings_separate_by_user
import numpy as np
def generate_feature_bins(feature_data, bins = 10): 
    sorted_data = np.sort(feature_data)
    bin_edges = np.percentile(sorted_data, np.linspace(0, 100, bins +1))
    return bin_edges

def generate_subvector_for_bins_and_value(bins, value):
    return np.array([1 if (value >= bins[i] and value < bins[i+1]) or (value == bins[-1] and bins[i+1] == bins[-1]) else 0 for i in range(bins.shape[0]-1)])

def generate_name_subvector(bins,name):
    return np.array([f"{bins[i]} <= {name} <{'=' if i+2 == bins.shape[0] else ''} {bins[i+1]}" for i in range(bins.shape[0]-1)])

def add_binned_features_to_vector(feature_vector, bin_edges, subvector_func):
    return np.concatenate((
        subvector_func(bin_edges[0], feature_vector[0]),
        feature_vector[1:11],
        subvector_func(bin_edges[12], feature_vector[12]),
        subvector_func(bin_edges[13], feature_vector[13]),
        subvector_func(bin_edges[14], feature_vector[14]),
        feature_vector[15:38],
        np.array([feature_vector[39]]) if feature_vector[39] == 'spoken_languages_count' else np.array([1 if feature_vector[39] > 1 else 0]),
        feature_vector[40:]
    ))

if __name__ == '__main__':
    numerical_values = {0:'budget',12:'popularity',13:'release_date',14:'runtime'}
    numerical_values_edges = dict()
    feature_vectors = load_movie_feature_vectors(True)
    for index, name in numerical_values.items():
        numerical_values_edges[index] = generate_feature_bins(feature_vectors[:,0])
    feature_vectors_names = add_binned_features_to_vector(load_movie_feature_vector_names(True), numerical_values_edges, generate_name_subvector)
    new_features = np.empty((0, feature_vectors_names.shape[0]), dtype=int)
    for i in range(feature_vectors.shape[0]):
        new_feature_vector = add_binned_features_to_vector(feature_vectors[i], numerical_values_edges, generate_subvector_for_bins_and_value)
        new_features = np.concatenate((new_features,new_feature_vector.reshape(1, -1)), axis = 0)
    np.savetxt('trees/binned_movie_feature_vector.csv',new_features.astype(int), delimiter=',', fmt='%d', header=','.join(feature_vectors_names), comments = '')
    generate_features_and_ratings_separate_by_user(new_features.astype(int),'trees/separated_binned_feature_vectors')
