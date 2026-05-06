import numpy as np
from preprocessing import preprocess_mnist

print("I am alive!", flush=True)

X_train, y_train, X_val, y_val, X_test, y_test = preprocess_mnist(
    method="pca",
    target_digit=5,
    pca_components=50
)

print("Data loaded successfully!", flush=True)
print("X_train:", X_train.shape, flush=True)
print("X_val:", X_val.shape, flush=True)
print("X_test:", X_test.shape, flush=True)


class KNNFromScratch:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []

        for i, x in enumerate(X):
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            k_indices = np.argsort(distances)[:self.k]

            k_nearest_labels = self.y_train[k_indices].astype(int)

            prediction = np.bincount(k_nearest_labels).argmax()

            predictions.append(prediction)

            if (i + 1) % 100 == 0:
                print(f"Predicted {i + 1}/{len(X)} samples...", flush=True)

        return np.array(predictions)


def confusion_matrix_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tn, fp], [fn, tp]])


def accuracy_score_manual(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_score_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score_manual(y_true, y_pred):
    p = precision_score_manual(y_true, y_pred)
    r = recall_score_manual(y_true, y_pred)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


print("Starting KNN...", flush=True)

knn = KNNFromScratch(k=3)

# Use all training samples
knn.fit(X_train, y_train)

print("Starting predictions on full test set...", flush=True)

# Use full test set: 10,000 samples
y_test_pred = knn.predict(X_test)
y_test_small = y_test

print("\n--- RESULTS ---", flush=True)
print("Accuracy :", accuracy_score_manual(y_test_small, y_test_pred), flush=True)
print("Precision:", precision_score_manual(y_test_small, y_test_pred), flush=True)
print("Recall   :", recall_score_manual(y_test_small, y_test_pred), flush=True)
print("F1-score :", f1_score_manual(y_test_small, y_test_pred), flush=True)
print("\nConfusion Matrix:\n", confusion_matrix_manual(y_test_small, y_test_pred), flush=True)