import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

from shared_code.helpers import load_user_feature_vector_from_file

def correlation_filter(x, y, corr_threshold = 0.15):
    correlations = np.array([np.corrcoef(x[:, i], y)[0, 1] for i in range(x.shape[1])])
    selected_indices = np.where(abs(correlations) > corr_threshold)[0]
    return x[:, selected_indices]

if __name__ == '__main__':
    for file in glob.glob(os.path.join('separated_feature_vectors','*.csv')):
        x, y, movie_ids = load_user_feature_vector_from_file(file)
        correlation_filter(x,y)
        # x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
        break