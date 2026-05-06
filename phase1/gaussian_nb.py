import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-8):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        n_samples, n_features = X.shape

        self.class_prior_ = {}
        self.theta_ = {}
        self.sigma_ = {}

        for cls, cnt in zip(self.classes_, counts):
            Xc = X[y == cls]
            self.class_prior_[cls] = cnt / n_samples
            self.theta_[cls] = Xc.mean(axis=0)
            self.sigma_[cls] = Xc.var(axis=0) + self.var_smoothing

        return self

    def _joint_log_likelihood(self, X):
        n_samples = X.shape[0]
        result = np.zeros((n_samples, len(self.classes_)))

        for idx, cls in enumerate(self.classes_):
            mean = self.theta_[cls]
            var = self.sigma_[cls]
            const = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            quad = -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            result[:, idx] = const + quad + np.log(self.class_prior_[cls])

        return result

    def predict(self, X):
        ll = self._joint_log_likelihood(X)
        idx = np.argmax(ll, axis=1)
        return self.classes_[idx]


def evaluate(y_true, y_pred, pos_label=1):
    acc = np.mean(y_true == y_pred)
    neg_label = [l for l in np.unique(y_true) if l != pos_label][0]
    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    tn = int(np.sum((y_true == neg_label) & (y_pred == neg_label)))
    fp = int(np.sum((y_true == neg_label) & (y_pred == pos_label)))
    fn = int(np.sum((y_true == pos_label) & (y_pred == neg_label)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
