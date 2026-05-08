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
    def __init__(self, k=3):
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

            # Faster than full sorting
            k_indices = np.argpartition(distances, self.k)[:self.k]

            k_nearest_labels = self.y_train[k_indices]

            # Majority vote for multiclass labels 0-9
            prediction = np.bincount(k_nearest_labels, minlength=10).argmax()
            predictions.append(prediction)

            if (i + 1) % 100 == 0:
                print(f"Predicted {i + 1}/{len(X)} samples...", flush=True)

        return np.array(predictions, dtype=int)


# ==========================================================
# MULTICLASS METRICS
# ==========================================================

def accuracy_score_manual(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix_multiclass(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    return cm


def classification_report_multiclass(y_true, y_pred, n_classes=10):
    cm = confusion_matrix_multiclass(y_true, y_pred, n_classes=n_classes)

    per_class = []
    precisions = []
    recalls = []
    f1s = []
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

        per_class.append({
            "class": c,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        })

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    total_support = np.sum(supports)

    weighted_precision = np.sum(np.array(precisions) * np.array(supports)) / total_support
    weighted_recall = np.sum(np.array(recalls) * np.array(supports)) / total_support
    weighted_f1 = np.sum(np.array(f1s) * np.array(supports)) / total_support

    return {
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm
    }


def print_multiclass_results(title, y_true, y_pred, n_classes=10):
    acc = accuracy_score_manual(y_true, y_pred)
    report = classification_report_multiclass(y_true, y_pred, n_classes=n_classes)

    print(f"\n--- {title} ---")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Macro Precision   : {report['macro_precision']:.4f}")
    print(f"Macro Recall      : {report['macro_recall']:.4f}")
    print(f"Macro F1-score    : {report['macro_f1']:.4f}")
    print(f"Weighted Precision: {report['weighted_precision']:.4f}")
    print(f"Weighted Recall   : {report['weighted_recall']:.4f}")
    print(f"Weighted F1-score : {report['weighted_f1']:.4f}")

    print("\nPer-class results:")
    print(f"{'Class':<8}{'Precision':<12}{'Recall':<12}{'F1-score':<12}{'Support':<10}")

    for row in report["per_class"]:
        print(
            f"{row['class']:<8}"
            f"{row['precision']:<12.4f}"
            f"{row['recall']:<12.4f}"
            f"{row['f1']:<12.4f}"
            f"{row['support']:<10d}"
        )

    print("\nConfusion Matrix:")
    print(report["confusion_matrix"])

    return acc, report


# ==========================================================
# RUN KNN FOR PCA, FLATTEN, AND HOG
# ==========================================================

def run_knn_experiment(method, k=3, pca_components=75, val_ratio=0.15):
    print("\n" + "=" * 70)
    print(f"Running KNN with method = {method}, k = {k}")
    print("=" * 70)

    if method == "pca":
        (X_train, y_train, X_val, y_val, X_test, y_test,
         mean, std, x_train_raw, x_val_raw, x_test_raw) = preprocess_mnist_multiclass(
            method="pca",
            pca_components=pca_components,
            val_ratio=val_ratio
        )
    else:
        (X_train, y_train, X_val, y_val, X_test, y_test,
         mean, std, x_train_raw, x_val_raw, x_test_raw) = preprocess_mnist_multiclass(
            method=method,
            val_ratio=val_ratio
        )

    print("Data loaded successfully!", flush=True)
    print("X_train shape:", X_train.shape, flush=True)
    print("X_val shape  :", X_val.shape, flush=True)
    print("X_test shape :", X_test.shape, flush=True)
    print("Classes      :", np.unique(y_train), flush=True)

    knn = KNNVectorized(k=k)
    knn.fit(X_train, y_train)

    print(f"\nRunning KNN on validation set using {method} features...", flush=True)
    y_val_pred = knn.predict(X_val)

    acc, report = print_multiclass_results(
        f"VALIDATION RESULTS — {method.upper()} — K={k}",
        y_val,
        y_val_pred,
        n_classes=10
    )

    return {
        "method": method,
        "k": k,
        "accuracy": acc,
        "macro_f1": report["macro_f1"],
        "weighted_f1": report["weighted_f1"]
    }


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    k = 3
    pca_components = 75

    methods = ["pca", "flatten", "hog"]

    summary_results = []

    for method in methods:
        result = run_knn_experiment(
            method=method,
            k=k,
            pca_components=pca_components,
            val_ratio=0.15
        )
        summary_results.append(result)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)

    print(f"{'Method':<12}{'K':<6}{'Accuracy':<12}{'Macro F1':<12}{'Weighted F1':<12}")

    for result in summary_results:
        print(
            f"{result['method']:<12}"
            f"{result['k']:<6}"
            f"{result['accuracy']:<12.4f}"
            f"{result['macro_f1']:<12.4f}"
            f"{result['weighted_f1']:<12.4f}"
        )