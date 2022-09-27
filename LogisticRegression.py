import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(x))


class LogisticRegression():

    def __init__(self, lr = .001, n_iters = 1000, threshold = 0.5):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.threshold = threshold
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            linear_prediction = np.dot(X, self.weights) + self.bias
            preds = sigmoid(linear_prediction)

            dw = (1/n_samples) * np.dot(X.T, (preds - y)) + self.bias
            db = (1/n_samples) * np.sum(preds - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    
    def predict(self, X):
        linear_prediction = np.dot(X, self.weights) + self.bias
        pred = sigmoid(linear_prediction)
        return [0 if i < self.threshold else 1 for i in pred]