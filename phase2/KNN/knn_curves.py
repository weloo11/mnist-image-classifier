import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from preprocessing2 import preprocess_mnist_multiclass


# ==========================================================
# 1) MULTICLASS VECTORIZED KNN
# ==========================================================
class KNNVectorized:
    """
    Multiclass KNN using Euclidean distance + majority voting.
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
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # Select k nearest neighbors
            k_indices = np.argpartition(distances, self.k - 1)[:self.k]
            k_labels = self.y_train[k_indices]

            # Majority vote among class labels 0..9
            pred = np.bincount(k_labels, minlength=10).argmax()
            predictions.append(pred)

        return np.array(predictions, dtype=int)


# ==========================================================
# 2) BASIC FEATURE HELPERS
# ==========================================================
def flatten_features(images):
    """
    Convert (N, 28, 28) images to (N, 784) vectors.
    """
    return images.reshape(images.shape[0], -1).astype(np.float32)


def fit_transform_pca_train_val(x_train_raw, x_val_raw, pca_components=50):
    """
    Fit PCA and standardization on train only,
    then transform validation using the same fitted objects.
    This avoids data leakage.
    """
    # Flatten first
    X_train_flat = flatten_features(x_train_raw)
    X_val_flat = flatten_features(x_val_raw)

    # Fit PCA on train only
    pca = PCA(n_components=pca_components)
    X_train = pca.fit_transform(X_train_flat)
    X_val = pca.transform(X_val_flat)

    # Standardize using train statistics only
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)

    return X_train, X_val


# ==========================================================
# 3) MULTICLASS METRICS
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
      - macro precision / recall / F1
      - weighted precision / recall / F1
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

    macro_precision = np.mean(per_class_precision)
    macro_recall = np.mean(per_class_recall)
    macro_f1 = np.mean(per_class_f1)

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
        "confusion_matrix": cm,
    }


# ==========================================================
# 4) LOAD RAW MULTICLASS DATA
# ==========================================================
# We only really need the raw images here because we want to
# fit PCA fresh for each experiment without leakage.
(
    X_train_dummy, y_train,
    X_val_dummy, y_val,
    X_test_dummy, y_test,
    mean, std,
    x_train_raw, x_val_raw, x_test_raw
) = preprocess_mnist_multiclass(
    method="pca",         # placeholder only
    pca_components=50,    # placeholder only
    val_ratio=0.15
)

print("Raw train shape:", x_train_raw.shape)
print("Raw val shape  :", x_val_raw.shape)
print("Raw test shape :", x_test_raw.shape)
print("Unique classes :", np.unique(y_train))


# ==========================================================
# 5) IMPROVEMENT A — LEARNING CURVE
# ==========================================================
def run_learning_curve(
    x_train_raw,
    y_train,
    x_val_raw,
    y_val,
    pca_components=50,
    k=1,
    train_fractions=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    random_state=42
):
    """
    Measures how train and validation performance change
    as the amount of training data increases.

    Interpretation:
      - high train, lower val => overfitting / high variance
      - both low => underfitting / high bias
      - val improves with more data => more data helps
    """
    rng = np.random.default_rng(random_state)

    train_sizes = []
    train_accuracies = []
    val_accuracies = []

    train_macro_f1s = []
    val_macro_f1s = []

    n_total = len(y_train)

    for frac in train_fractions:
        n_subset = int(n_total * frac)

        # Sample a subset of the training data
        indices = rng.permutation(n_total)[:n_subset]
        x_train_subset = x_train_raw[indices]
        y_train_subset = y_train[indices]

        # Fit PCA + scaling on this subset only
        X_train_proc, X_val_proc = fit_transform_pca_train_val(
            x_train_subset,
            x_val_raw,
            pca_components=pca_components
        )

        # Also transform the same train subset for train score
        X_train_subset_proc, _ = fit_transform_pca_train_val(
            x_train_subset,
            x_train_subset,
            pca_components=pca_components
        )

        model = KNNVectorized(k=k)
        model.fit(X_train_proc, y_train_subset)

        y_train_pred = model.predict(X_train_subset_proc)
        y_val_pred = model.predict(X_val_proc)

        train_metrics = multiclass_metrics(y_train_subset, y_train_pred)
        val_metrics = multiclass_metrics(y_val, y_val_pred)

        train_sizes.append(n_subset)
        train_accuracies.append(train_metrics["accuracy"])
        val_accuracies.append(val_metrics["accuracy"])

        train_macro_f1s.append(train_metrics["macro_f1"])
        val_macro_f1s.append(val_metrics["macro_f1"])

        print(
            f"Train size={n_subset:5d} | "
            f"Train Acc={train_metrics['accuracy']:.4f} | "
            f"Val Acc={val_metrics['accuracy']:.4f} | "
            f"Train MacroF1={train_metrics['macro_f1']:.4f} | "
            f"Val MacroF1={val_metrics['macro_f1']:.4f}"
        )

    return {
        "train_sizes": train_sizes,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "train_macro_f1s": train_macro_f1s,
        "val_macro_f1s": val_macro_f1s,
    }


# ==========================================================
# 6) IMPROVEMENT B — VALIDATION CURVE OVER K
# ==========================================================
def run_validation_curve_over_k(
    x_train_raw,
    y_train,
    x_val_raw,
    y_val,
    pca_components=50,
    k_values=(1, 3, 5, 7, 9, 11)
):
    """
    Measures training and validation performance as k changes.

    Interpretation:
      - very small k => lower bias, higher variance
      - larger k => higher bias, lower variance
      - best validation point => best tradeoff
    """
    train_accuracies = []
    val_accuracies = []

    train_macro_f1s = []
    val_macro_f1s = []

    # Fit PCA + scaling once using the full train set,
    # then apply to validation.
    X_train_proc, X_val_proc = fit_transform_pca_train_val(
        x_train_raw,
        x_val_raw,
        pca_components=pca_components
    )

    # Also get train->train transformed features for train score
    X_train_train_proc, _ = fit_transform_pca_train_val(
        x_train_raw,
        x_train_raw,
        pca_components=pca_components
    )

    for k in k_values:
        model = KNNVectorized(k=k)
        model.fit(X_train_proc, y_train)

        y_train_pred = model.predict(X_train_train_proc)
        y_val_pred = model.predict(X_val_proc)

        train_metrics = multiclass_metrics(y_train, y_train_pred)
        val_metrics = multiclass_metrics(y_val, y_val_pred)

        train_accuracies.append(train_metrics["accuracy"])
        val_accuracies.append(val_metrics["accuracy"])

        train_macro_f1s.append(train_metrics["macro_f1"])
        val_macro_f1s.append(val_metrics["macro_f1"])

        print(
            f"k={k:2d} | "
            f"Train Acc={train_metrics['accuracy']:.4f} | "
            f"Val Acc={val_metrics['accuracy']:.4f} | "
            f"Train MacroF1={train_metrics['macro_f1']:.4f} | "
            f"Val MacroF1={val_metrics['macro_f1']:.4f}"
        )

    return {
        "k_values": list(k_values),
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "train_macro_f1s": train_macro_f1s,
        "val_macro_f1s": val_macro_f1s,
    }


# ==========================================================
# 7) PLOT HELPERS
# ==========================================================
def plot_learning_curve(results):
    """
    Plots training vs validation performance as training size grows.
    """
    plt.figure(figsize=(9, 5))
    plt.plot(results["train_sizes"], results["train_accuracies"], marker="o", label="Train Accuracy")
    plt.plot(results["train_sizes"], results["val_accuracies"], marker="o", label="Validation Accuracy")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve — Accuracy vs Training Set Size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(results["train_sizes"], results["train_macro_f1s"], marker="o", label="Train Macro F1")
    plt.plot(results["train_sizes"], results["val_macro_f1s"], marker="o", label="Validation Macro F1")
    plt.xlabel("Training Set Size")
    plt.ylabel("Macro F1-score")
    plt.title("Learning Curve — Macro F1 vs Training Set Size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_validation_curve(results):
    """
    Plots training vs validation performance as k changes.
    """
    plt.figure(figsize=(9, 5))
    plt.plot(results["k_values"], results["train_accuracies"], marker="o", label="Train Accuracy")
    plt.plot(results["k_values"], results["val_accuracies"], marker="o", label="Validation Accuracy")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Validation Curve — Accuracy vs k")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(results["k_values"], results["train_macro_f1s"], marker="o", label="Train Macro F1")
    plt.plot(results["k_values"], results["val_macro_f1s"], marker="o", label="Validation Macro F1")
    plt.xlabel("k")
    plt.ylabel("Macro F1-score")
    plt.title("Validation Curve — Macro F1 vs k")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================================
# 8) RUN THE THIRD IMPROVEMENT
# ==========================================================
if __name__ == "__main__":
    # Current best tuned baseline from CV
    best_pca_components = 50
    best_k = 1

    print("\n" + "=" * 70)
    print("LEARNING CURVE ANALYSIS")
    print("=" * 70)
    learning_curve_results = run_learning_curve(
        x_train_raw=x_train_raw,
        y_train=y_train,
        x_val_raw=x_val_raw,
        y_val=y_val,
        pca_components=best_pca_components,
        k=best_k,
        train_fractions=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
        random_state=42
    )

    print("\n" + "=" * 70)
    print("VALIDATION CURVE OVER k")
    print("=" * 70)
    validation_curve_results = run_validation_curve_over_k(
        x_train_raw=x_train_raw,
        y_train=y_train,
        x_val_raw=x_val_raw,
        y_val=y_val,
        pca_components=best_pca_components,
        k_values=(1, 3, 5, 7, 9, 11)
    )

    # Plot the results
    plot_learning_curve(learning_curve_results)
    plot_validation_curve(validation_curve_results)