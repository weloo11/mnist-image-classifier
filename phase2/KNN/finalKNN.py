import numpy as np
from sklearn.decomposition import PCA
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from preprocessing2 import preprocess_mnist_multiclass
# ==========================================================
# 1) FINAL SELECTED MODEL SETTINGS
# ==========================================================
BEST_K = 1
BEST_PCA_COMPONENTS = 50
VAL_RATIO = 0.15


# ==========================================================
# 2) VECTORIZED MULTICLASS KNN
# ==========================================================
class KNNVectorized:
    """
    Multiclass KNN using Euclidean distance and majority voting.
    """
    def __init__(self, k=1):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.astype(np.float32)
        self.y_train = y.astype(int)

    def predict(self, X):
        X = X.astype(np.float32)
        predictions = []

        for i, x in enumerate(X):
            # Distance from x to every training sample
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # Get indices of k nearest neighbors
            k_indices = np.argpartition(distances, self.k - 1)[:self.k]

            # Get labels of those neighbors
            k_labels = self.y_train[k_indices]

            # Majority vote
            pred = np.bincount(k_labels, minlength=10).argmax()
            predictions.append(pred)

            if (i + 1) % 100 == 0:
                print(f"Predicted {i + 1}/{len(X)} test samples...", flush=True)

        return np.array(predictions, dtype=int)


# ==========================================================
# 3) FEATURE HELPERS
# ==========================================================
def flatten_features(images):
    """
    Convert each 28x28 image into a 784-dimensional vector.
    """
    return images.reshape(images.shape[0], -1).astype(np.float32)


def fit_pca_and_standardize(x_dev_raw, x_test_raw, pca_components=50):
    """
    Fit PCA and standardization on the full development set only,
    then transform the test set using the same fitted preprocessing.

    This is the correct final-evaluation procedure:
    - development set = train + validation
    - test set remains untouched until the end
    """
    # Flatten raw images
    X_dev_flat = flatten_features(x_dev_raw)
    X_test_flat = flatten_features(x_test_raw)

    # Fit PCA on development data only
    pca = PCA(n_components=pca_components)
    X_dev = pca.fit_transform(X_dev_flat)
    X_test = pca.transform(X_test_flat)

    # Standardize using development statistics only
    mean = np.mean(X_dev, axis=0)
    std = np.std(X_dev, axis=0) + 1e-8

    X_dev = ((X_dev - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    return X_dev, X_test


# ==========================================================
# 4) MULTICLASS METRICS
# ==========================================================
def confusion_matrix_multiclass(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    return cm


def multiclass_metrics(y_true, y_pred, n_classes=10):
    """
    Computes:
    - accuracy
    - macro precision / recall / f1
    - weighted precision / recall / f1
    - confusion matrix
    """
    cm = confusion_matrix_multiclass(y_true, y_pred, n_classes=n_classes)

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    supports = []

    for c in range(n_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        support = np.sum(cm[c, :])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
        supports.append(support)

    per_class_precision = np.array(per_class_precision, dtype=np.float64)
    per_class_recall = np.array(per_class_recall, dtype=np.float64)
    per_class_f1 = np.array(per_class_f1, dtype=np.float64)
    supports = np.array(supports, dtype=np.float64)

    accuracy = np.mean(y_true == y_pred)

    # Macro averages
    macro_precision = np.mean(per_class_precision)
    macro_recall = np.mean(per_class_recall)
    macro_f1 = np.mean(per_class_f1)

    # Weighted averages
    total_support = np.sum(supports)
    weighted_precision = np.sum(per_class_precision * supports) / total_support
    weighted_recall = np.sum(per_class_recall * supports) / total_support
    weighted_f1 = np.sum(per_class_f1 * supports) / total_support

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm
    }


def print_results(title, y_true, y_pred, n_classes=10):
    results = multiclass_metrics(y_true, y_pred, n_classes=n_classes)

    print(f"\n--- {title} ---")
    print(f"Accuracy            : {results['accuracy']:.4f}")
    print(f"Macro Precision     : {results['macro_precision']:.4f}")
    print(f"Macro Recall        : {results['macro_recall']:.4f}")
    print(f"Macro F1-score      : {results['macro_f1']:.4f}")
    print(f"Weighted Precision  : {results['weighted_precision']:.4f}")
    print(f"Weighted Recall     : {results['weighted_recall']:.4f}")
    print(f"Weighted F1-score   : {results['weighted_f1']:.4f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    return results


# ==========================================================
# 5) LOAD DATA
# ==========================================================
# We use the multiclass preprocessing function only to get:
# - y_train, y_val, y_test
# - raw images (x_train_raw, x_val_raw, x_test_raw)
#
# We do NOT use the already-processed PCA outputs here,
# because for the final model we want to fit PCA again on the
# full development set (train + val) only.
(
    X_train_dummy, y_train,
    X_val_dummy, y_val,
    X_test_dummy, y_test,
    mean, std,
    x_train_raw, x_val_raw, x_test_raw
) = preprocess_mnist_multiclass(
    method="flatten",                 # placeholder
    pca_components=BEST_PCA_COMPONENTS,
    val_ratio=VAL_RATIO
)

print("Raw train shape:", x_train_raw.shape)
print("Raw val shape  :", x_val_raw.shape)
print("Raw test shape :", x_test_raw.shape)
print("Classes        :", np.unique(y_train))


# ==========================================================
# 6) BUILD FULL DEVELOPMENT SET
# ==========================================================
# Final training should use train + validation together
x_dev_raw = np.concatenate([x_train_raw, x_val_raw], axis=0)
y_dev = np.concatenate([y_train, y_val], axis=0)

print("\nFull development set shape:", x_dev_raw.shape)
print("Full test set shape       :", x_test_raw.shape)


# ==========================================================
# 7) FIT PCA(50) + STANDARDIZE ON DEVELOPMENT SET
# ==========================================================
X_dev, X_test = fit_pca_and_standardize(
    x_dev_raw=x_dev_raw,
    x_test_raw=x_test_raw,
    pca_components=BEST_PCA_COMPONENTS
)

print("\nProcessed development shape:", X_dev.shape)
print("Processed test shape       :", X_test.shape)


# ==========================================================
# 8) TRAIN FINAL KNN MODEL
# ==========================================================
print(f"\nTraining final KNN model with k={BEST_K} and PCA={BEST_PCA_COMPONENTS}...", flush=True)

final_knn = KNNVectorized(k=BEST_K)
final_knn.fit(X_dev, y_dev)


# ==========================================================
# 9) FINAL TEST EVALUATION
# ==========================================================
print("\nRunning final evaluation on the untouched test set...", flush=True)

y_test_pred = final_knn.predict(X_test)

final_results = print_results(
    title=f"FINAL TEST RESULTS — PCA({BEST_PCA_COMPONENTS}) + KNN(k={BEST_K})",
    y_true=y_test,
    y_pred=y_test_pred,
    n_classes=10
)
