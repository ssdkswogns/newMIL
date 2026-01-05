import numpy as np

class ED1NN:
    """
    Euclidean Distance 1-NN classifier.
    X: (n_samples, n_features) 또는 (n_samples, T, C) 같은 시계열도 가능(reshape로 처리)
    """
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.X_train = X.reshape(X.shape[0], -1).astype(np.float32)
        self.y_train = y
        return self

    def predict(self, X):
        X = np.asarray(X).reshape(np.asarray(X).shape[0], -1).astype(np.float32)

        # (n_test, n_train) 거리 행렬: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        X2 = np.sum(X * X, axis=1, keepdims=True)              # (n_test, 1)
        T2 = np.sum(self.X_train * self.X_train, axis=1)[None] # (1, n_train)
        d2 = X2 + T2 - 2.0 * (X @ self.X_train.T)              # (n_test, n_train)
        nn_idx = np.argmin(d2, axis=1)
        return self.y_train[nn_idx]

    def score(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))


# 사용 예시
if __name__ == "__main__":
    X_train = np.array([[0, 0], [1, 1], [10, 10]])
    y_train = np.array([0, 0, 1])
    X_test = np.array([[0.2, 0.1], [9, 9]])

    clf = ED1NN().fit(X_train, y_train)
    print(clf.predict(X_test))  # [0 1]
