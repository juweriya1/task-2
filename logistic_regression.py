import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit  

class LogisticRegression:

    def __init__(self, learning_rate=0.001, number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        number_of_samples, number_of_features = X.shape
        self.weights = np.zeros(number_of_features)
        self.bias = 0

        for _ in range(self.number_of_iterations):
            linear_model = X.dot(self.weights) + self.bias
            if isinstance(linear_model, csr_matrix):
                linear_model = linear_model.toarray()
            predicted_y = expit(linear_model)
            dw = (1 / number_of_samples) * X.T.dot(predicted_y - y)
            db = (1 / number_of_samples) * np.sum(predicted_y - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = X.dot(self.weights) + self.bias
        if isinstance(linear_model, csr_matrix):
            linear_model = linear_model.toarray()
        predicted_y = expit(linear_model)
        return [1 if i > 0.5 else 0 for i in predicted_y]
    
