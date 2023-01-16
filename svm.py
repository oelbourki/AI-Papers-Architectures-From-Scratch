import numpy as np

class SVM:
    def __init__(self, lr=0.001, lmd=0.01, n_iters=1000) -> None:
        self.lr = lr
        self.lmd = lmd
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(self.w, X) - self.b) >= 1
                if condition:
                    self.w += self.lr * (2 * self.lmd * self.w)
                else:
                    self.w += self.lr * (2 * self.lmd * self.w - np.dot(x_i, y_[idx]))
                    self.b = self.lr * y_[idx]
    
    def predict(self, X):
        res = np.dot(X, self.w) - self.b
        return np.sign(res)