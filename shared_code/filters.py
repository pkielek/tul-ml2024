import numpy as np
from sklearn.feature_selection import mutual_info_regression

def correlation_filter(x, y, corr_threshold = 0.1):
    correlations = np.array([np.corrcoef(x[:, i], y)[0, 1] for i in range(x.shape[1])])
    selected_indices = np.where(abs(correlations) > corr_threshold)[0]
    return selected_indices

def mutual_information_filter(x, y, mi_threshold=0.05):
    mi_scores = mutual_info_regression(x, y)
    selected_indices = np.where(mi_scores > mi_threshold)[0]
    return selected_indices

def greedy_backwards_feature_selection(current_accuracy, movie_features, selected_features_indices, train_and_predict):
    print(len(selected_features_indices))
    best_accuracy = current_accuracy
    best_indices = selected_features_indices
    for i, _ in enumerate(selected_features_indices):
        new_selected_features_indices = np.delete(selected_features_indices, i)
        new_accuracy = train_and_predict(movie_features[:, new_selected_features_indices])
        if new_accuracy > best_accuracy:
            best_indices = new_selected_features_indices
            best_accuracy = new_accuracy
    if best_accuracy > current_accuracy:
        return greedy_backwards_feature_selection(best_accuracy, movie_features, best_indices, train_and_predict)
    return best_accuracy, best_indices

# def greedy_forward_feature_selection(current_accuracy, movie_features, selected_features_indices, train_and_predict):
#     print(len(selected_features_indices))
#     best_accuracy = current_accuracy
#     best_indices = selected_features_indices
#     for i in np.arange(movie_features.shape[1]):
#         if i not in selected_features_indices:
#             new_selected_features_indices = np.append(selected_features_indices, i)
#             new_accuracy = train_and_predict(movie_features[:, new_selected_features_indices])
#             if new_accuracy > best_accuracy:
#                 best_indices = new_selected_features_indices
#                 best_accuracy = new_accuracy
#     if best_accuracy > current_accuracy:
#         return greedy_forward_feature_selection(best_accuracy, movie_features, best_indices, train_and_predict)
#     return best_accuracy, best_indices