from DecisionTrees import DecisionTree
import numpy as np
from collectioms import Counter

class RandomForest:
    def __init__(self, n_trees = 100, max_depth = 10, n_features = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.n_features = n_features

    def fit(self, X, y):
        for _ in range(self.n_trees):
            current_tree = DecisionTree(min_samples_split = self.min_samples_split,
                                        max_depth = self.max_depth,
                                        n_features = self.n_features)
            X_train, y_train = self.bootstrap(X, y)
            current_tree.fit(X_train, y_train)
            self.trees.append(current_tree)
        

    # select n samples from n the n total with chance of repetition
    def bootstrap(self, X, y):
        n_samples = X.shape[0]
        selected_samples = np.random.choice(n_samples, n_samples, replace = True)
        return X[selected_samples], y[selected_samples]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions


    