import numpy as np
from shared_code.helpers import get_user_list,get_train_data, get_task_data
from random import random


class CollaborativeFiltering:
    def __init__(self, train_data, user_list, n_features, lambda_reg, validation_data = None, n_epochs = 300, lr = 0.001):
        self.train_data = train_data
        self.user_list = user_list
        self.n_features = n_features
        self.lambda_reg = lambda_reg
        self.validation_data = validation_data
        self.n_epochs = n_epochs
        self.lr = lr

        self.movie_features = {i:np.random.normal(scale=1.0/n_features, size=n_features) for i in range(1,201)}
        self.user_features = {user_id:np.random.normal(scale=1.0/n_features, size=n_features) for user_id in user_list}
        self.bias = np.random.rand()
        self.user_movie_ratings = {user_id:dict() for user_id in user_list}
        for entry in train_data:
                user_id, movie_id, rating = entry[1], entry[2], entry[3]
                self.user_movie_ratings[user_id][movie_id] = rating
        self.user_means = {user_id: np.mean(list(self.user_movie_ratings[user_id].items())) for user_id in user_list}
        self.normalized_user_movie_ratings = {user_id:{movie_id:self.user_movie_ratings[user_id][movie_id] - self.user_means[user_id] for movie_id in self.user_movie_ratings[user_id]} for user_id in user_list}
        self.predictions = None

    def calc_loss(self):
        error_loss = 0.0
        for user_id in self.user_list:
            for movie_id in self.user_movie_ratings[user_id]:
                predicted_rating = np.dot(self.user_features[user_id], self.movie_features[movie_id]) + self.bias
                error_loss += (self.normalized_user_movie_ratings[user_id][movie_id] - predicted_rating) ** 2
        error_loss /= 2
        user_feature_norm = sum(np.linalg.norm(self.user_features[user_id])**2 for user_id in self.user_list)
        movie_feature_norm = sum(np.linalg.norm(self.movie_features[movie_id])**2 for movie_id in range(1, 201))
        regularization_loss = self.lambda_reg * (user_feature_norm + movie_feature_norm) / 2
        return regularization_loss + error_loss

    def calc_accuracy(self):
        correct = 0
        for entry in self.validation_data:
            user_id, movie_id, rating = entry[1], entry[2], entry[3]
            if rating == self.predictions[user_id][movie_id]: correct += 1
        return correct / len(self.validation_data)
    
    def calc_predictions(self):
        self.predictions = {user_id: {} for user_id in self.user_list}
        for user_id in self.user_list:
            for movie_id in range(1, 201):
                self.predictions[user_id][movie_id] = max(0, min(5, round(np.dot(self.user_features[user_id], self.movie_features[movie_id]) + self.bias + self.user_means[user_id])))

    def train(self):
        for epoch in range(self.n_epochs):
            for user_id in self.user_list:
                for movie_id in self.user_movie_ratings[user_id]:
                    error = self.normalized_user_movie_ratings[user_id][movie_id] - np.dot(self.user_features[user_id], self.movie_features[movie_id]) - self.bias
                    self.user_features[user_id] += self.lr * (error * self.movie_features[movie_id] - self.lambda_reg * self.user_features[user_id])
                    self.movie_features[movie_id] += self.lr * (error * self.user_features[user_id] - self.lambda_reg * self.movie_features[movie_id])
                    self.bias += self.lr * error
            self.calc_predictions()
            print(f"Epoch {epoch + 1}, Loss: {self.calc_loss()}, Accuracy: {'null' if self.validation_data is None else self.calc_accuracy()}")
        if self.validation_data is not None:
            self.calc_predictions()
            print(f"Final accuracy:{self.calc_accuracy()}")

    def classify(self, task_data, filename = 'submission'):
        task_results = np.empty((0, 4), dtype=int)
        print("Classifying")
        for row in task_data:
            user_id, movie_id = row[1], row[2]
            task_results = np.append(task_results, [np.array([row[0], row[1], row[2], self.predictions[user_id][movie_id]], dtype=int)], axis=0)
        np.savetxt(f'collaborative/{filename}.csv', task_results, delimiter=';', fmt="%d")


lambda_reg = 0.02
n_features = 100
if __name__ == '__main__':
    task_data = get_task_data()
    train_data = get_train_data()
    user_list = get_user_list()
    if False: # Test
        split_train_data = []
        validation_data = []
        for entry in train_data:
            if random() <= 0.15:
                validation_data.append(entry)
            else:
                split_train_data.append(entry)
        user_list = get_user_list()
        algorithm = CollaborativeFiltering(split_train_data,user_list,n_features,lambda_reg,validation_data)
        algorithm.train()
    else: # Classify
        for n in [200,300,500,1000]:
            classify_algorithm = CollaborativeFiltering(train_data,user_list,n_features,lambda_reg,n_epochs=n)
            classify_algorithm.train()
            classify_algorithm.classify(task_data,f"submission{n}")
