import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from preprocessing2 import preprocess_mnist_multiclass


# ==========================================================
# VECTORIZED KNN — MULTICLASS
# ==========================================================
class KNNVectorized:
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
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # Faster nearest-neighbor selection
            k_indices = np.argpartition(distances, self.k - 1)[:self.k]

            k_nearest_labels = self.y_train[k_indices]

            # Majority vote for labels 0-9
            prediction = np.bincount(k_nearest_labels, minlength=10).argmax()
            predictions.append(prediction)

            if (i + 1) % 100 == 0:
                print(f"Predicted {i + 1}/{len(X)} samples...", flush=True)

        return np.array(predictions, dtype=int)


# ==========================================================
# MULTICLASS METRICS
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

    # Convert to arrays for averaging
    per_class_precision = np.array(per_class_precision, dtype=np.float64)
    per_class_recall = np.array(per_class_recall, dtype=np.float64)
    per_class_f1 = np.array(per_class_f1, dtype=np.float64)
    supports = np.array(supports, dtype=np.float64)

    # Accuracy
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
# RUN THE NEW PCA BASELINE
# ==========================================================
def run_knn_baseline(k=1, pca_components=50, val_ratio=0.15):
    print("\n" + "=" * 70)
    print(f"Running tuned multiclass KNN baseline: PCA, k={k}, pca_components={pca_components}")
    print("=" * 70)

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        mean, std,
        x_train_raw, x_val_raw, x_test_raw
    ) = preprocess_mnist_multiclass(
        method="pca",
        pca_components=pca_components,
        val_ratio=val_ratio
    )

    print("Data loaded successfully!", flush=True)
    print("X_train shape:", X_train.shape, flush=True)
    print("X_val shape  :", X_val.shape, flush=True)
    print("X_test shape :", X_test.shape, flush=True)
    print("Classes      :", np.unique(y_train), flush=True)

    knn = KNNVectorized(k=k)
    knn.fit(X_train, y_train)

    print("\nRunning tuned KNN baseline on validation set...", flush=True)
    y_val_pred = knn.predict(X_val)

    results = print_results(
        title=f"VALIDATION RESULTS — PCA({pca_components}) — K={k}",
        y_true=y_val,
        y_pred=y_val_pred,
        n_classes=10
    )

    return results


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    # New tuned baseline from 5-fold CV
    k = 1
    pca_components = 50

    run_knn_baseline(
        k=k,
        pca_components=pca_components,
        val_ratio=0.15
    )