import numpy as np
from preprocessing import preprocess_mnist


# ==========================================================
# PREPROCESSING
# ==========================================================
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_mnist(
    method="pca",
    target_digit=5,
    pca_components=75
)

print("Data loaded successfully!", flush=True)
print("X_train shape:", X_train.shape, flush=True)
print("X_val shape  :", X_val.shape, flush=True)
print("X_test shape :", X_test.shape, flush=True)


# ==========================================================
# KNN LOOP VERSION
# ==========================================================
class KNNLoop:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict_one(self, x):
        distances = []

        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda pair: pair[0])
        k_nearest_labels = [label for _, label in distances[:self.k]]

        prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return prediction

    def predict(self, X):
        predictions = []

        for i, x in enumerate(X):
            pred = self.predict_one(x)
            predictions.append(pred)

            if (i + 1) % 50 == 0:
                print(f"[Loop] Predicted {i + 1}/{len(X)} samples...", flush=True)

        return np.array(predictions)


# ==========================================================
# KNN VECTORIZED VERSION
# ==========================================================
class KNNVectorized:
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
                print(f"[Vectorized] Predicted {i + 1}/{len(X)} samples...", flush=True)

        return np.array(predictions)


# ==========================================================
# METRICS
# ==========================================================
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
    return 0.0 if (tp + fp) == 0 else tp / (tp + fp)


def recall_score_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return 0.0 if (tp + fn) == 0 else tp / (tp + fn)


def f1_score_manual(y_true, y_pred):
    p = precision_score_manual(y_true, y_pred)
    r = recall_score_manual(y_true, y_pred)
    return 0.0 if (p + r) == 0 else 2 * (p * r) / (p + r)


def print_metrics(title, y_true, y_pred):
    print(f"\n--- {title} ---", flush=True)
    print("Accuracy :", accuracy_score_manual(y_true, y_pred), flush=True)
    print("Precision:", precision_score_manual(y_true, y_pred), flush=True)
    print("Recall   :", recall_score_manual(y_true, y_pred), flush=True)
    print("F1-score :", f1_score_manual(y_true, y_pred), flush=True)
    print("Confusion Matrix:\n", confusion_matrix_manual(y_true, y_pred), flush=True)


# ==========================================================
# STEP 1: COMPARE LOOP VS VECTORIZED ON 500 VALIDATION SAMPLES
# ==========================================================
subset_size = 500
X_val_subset = X_val[:subset_size]
y_val_subset = y_val[:subset_size]

print("\nComparing KNNLoop and KNNVectorized on validation subset...", flush=True)

knn_loop = KNNLoop(k=3)
knn_loop.fit(X_train, y_train)
y_val_pred_loop = knn_loop.predict(X_val_subset)

knn_vec = KNNVectorized(k=3)
knn_vec.fit(X_train, y_train)
y_val_pred_vec = knn_vec.predict(X_val_subset)

print_metrics("KNNLoop - Validation Subset", y_val_subset, y_val_pred_loop)
print_metrics("KNNVectorized - Validation Subset", y_val_subset, y_val_pred_vec)

same_predictions = np.array_equal(y_val_pred_loop, y_val_pred_vec)
print("\nAre loop and vectorized predictions exactly equal on the subset?", same_predictions, flush=True)


# ==========================================================
# STEP 2: IF EQUAL, USE VECTORIZED ON FULL VALIDATION + TEST
# ==========================================================
if same_predictions:
    print("\nLoop and vectorized versions are identical on the subset.", flush=True)
    print("Proceeding with vectorized KNN on full validation and test sets...", flush=True)

    final_knn = KNNVectorized(k=3)
    final_knn.fit(X_train, y_train)

    # Full validation
    print("\nRunning vectorized KNN on full validation set...", flush=True)
    y_val_pred_full = final_knn.predict(X_val)
    print_metrics("FINAL VALIDATION RESULTS", y_val, y_val_pred_full)

    # Full test
    print("\nRunning vectorized KNN on full test set...", flush=True)
    y_test_pred = final_knn.predict(X_test)
    print_metrics("FINAL TEST RESULTS", y_test, y_test_pred)

else:
    print("\nWARNING: Loop and vectorized predictions are NOT identical on the subset.", flush=True)
    print("Do not use the vectorized version for final evaluation until the implementations match.", flush=True)