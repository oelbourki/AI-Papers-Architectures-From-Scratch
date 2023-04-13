import numpy as np

class DecisionStump:
    def __init__(self) -> None:
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None
    
    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        
        return predictions

class Adaboost:
    def __init__(self, n_clf=5) -> None:
        self.n_clf = n_clf
        self.clfs = []
        
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1/n_samples))
        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float("inf")
            
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    
                    missclassified = w[y != predictions]
                    error = np.sum(missclassified)
                    
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1. - min_error + EPS) / (min_error + EPS))
            
            predictions = clf.predict(X)
            
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)
            self.clfs.append(clf)
    
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5
    )

    # Adaboost classification with 5 weak classifiers
    clf = Adaboost(n_clf=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy(y_test, y_pred)
    print("Accuracy:", acc)