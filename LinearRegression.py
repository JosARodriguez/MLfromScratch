import numpy as np

class LogisticRegression():

    def __init__(self, lr = .001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            preds = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (preds - y)) + self.bias
            db = (1/n_samples) * np.sum(preds - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    
    def predict(self, X):
        preds = np.dot(X, self.weights) + self.bias
        return np.round(preds)