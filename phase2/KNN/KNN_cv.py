import numpy as np
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
    Multiclass KNN classifier from scratch.
    Uses Euclidean distance and majority voting.
    """
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.astype(np.float32)
        self.y_train = y.astype(int)

    def predict(self, X):
        predictions = []

        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]

            pred = np.bincount(k_labels, minlength=10).argmax()
            predictions.append(pred)

        return np.array(predictions, dtype=int)


# ==========================================================
# 2) SIMPLE FLATTEN HELPER
# ==========================================================
def flatten_features(images):
    """
    Convert each 28x28 image into a 784-dimensional vector.
    """
    return images.reshape(images.shape[0], -1).astype(np.float32)


# ==========================================================
# 3) FIT PCA + STANDARDIZATION INSIDE EACH FOLD
# ==========================================================
def fit_transform_pca_fold(x_train_raw, x_val_raw, pca_components):
    """
    Fit PCA and scaling on fold-train only, then apply to fold-validation.
    This avoids data leakage.
    """
    X_train_flat = flatten_features(x_train_raw)
    X_val_flat = flatten_features(x_val_raw)

    pca = PCA(n_components=pca_components)
    X_train = pca.fit_transform(X_train_flat)
    X_val = pca.transform(X_val_flat)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)

    return X_train, X_val


# ==========================================================
# 4) MANUAL MULTICLASS METRICS
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
        "confusion_matrix": cm
    }


# ==========================================================
# 5) MANUAL STRATIFIED K-FOLD SPLIT
# ==========================================================
def stratified_kfold_indices(y, n_splits=5, random_state=42):
    """
    Manual replacement for StratifiedKFold.

    What it does:
    - separates indices by class
    - shuffles each class separately
    - splits each class into n_splits parts
    - builds each fold so that class proportions stay balanced
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)

    # For each class, build n_splits chunks
    class_fold_chunks = {}

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)

        # Split this class's indices into n_splits nearly equal chunks
        chunks = np.array_split(cls_indices, n_splits)
        class_fold_chunks[cls] = chunks

    # Build folds: fold i validation = chunk i from every class
    folds = []
    all_indices = np.arange(len(y))

    for fold_idx in range(n_splits):
        val_parts = [class_fold_chunks[cls][fold_idx] for cls in classes]
        val_idx = np.concatenate(val_parts)
        rng.shuffle(val_idx)

        val_mask = np.zeros(len(y), dtype=bool)
        val_mask[val_idx] = True
        train_idx = all_indices[~val_mask]

        folds.append((train_idx, val_idx))

    return folds


# ==========================================================
# 6) ONE FULL 5-FOLD CV RUN FOR ONE PARAMETER COMBINATION
# ==========================================================
def cross_validate_knn_pca(
    x_dev_raw,
    y_dev,
    k_neighbors=3,
    pca_components=75,
    n_splits=5,
    random_state=42
):
    """
    Run manual stratified 5-fold CV for one combination of:
      - k_neighbors
      - pca_components

    Returns mean/std of:
      - accuracy
      - macro F1
      - weighted F1
    """
    folds = stratified_kfold_indices(
        y=y_dev,
        n_splits=n_splits,
        random_state=random_state
    )

    fold_accuracies = []
    fold_macro_f1s = []
    fold_weighted_f1s = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        x_fold_train = x_dev_raw[train_idx]
        y_fold_train = y_dev[train_idx]

        x_fold_val = x_dev_raw[val_idx]
        y_fold_val = y_dev[val_idx]

        X_fold_train, X_fold_val = fit_transform_pca_fold(
            x_fold_train,
            x_fold_val,
            pca_components=pca_components
        )

        model = KNNVectorized(k=k_neighbors)
        model.fit(X_fold_train, y_fold_train)

        y_pred = model.predict(X_fold_val)
        metrics = multiclass_metrics(y_fold_val, y_pred, n_classes=10)

        acc = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]
        weighted_f1 = metrics["weighted_f1"]

        fold_accuracies.append(acc)
        fold_macro_f1s.append(macro_f1)
        fold_weighted_f1s.append(weighted_f1)

        print(
            f"Fold {fold_idx}: "
            f"Accuracy={acc:.4f}, "
            f"Macro F1={macro_f1:.4f}, "
            f"Weighted F1={weighted_f1:.4f}"
        )

    return {
        "accuracy_mean": np.mean(fold_accuracies),
        "accuracy_std": np.std(fold_accuracies),
        "macro_f1_mean": np.mean(fold_macro_f1s),
        "macro_f1_std": np.std(fold_macro_f1s),
        "weighted_f1_mean": np.mean(fold_weighted_f1s),
        "weighted_f1_std": np.std(fold_weighted_f1s),
    }


# ==========================================================
# 7) MAIN TUNING FUNCTION
# ==========================================================
def tune_knn_with_5fold_cv():
    """
    Full tuning workflow:
    1) Load multiclass MNIST
    2) Combine train + validation into one development set
    3) Tune KNN over:
         - k
         - pca_components
    4) Select best configuration using macro F1
    """
    (
        X_train_dummy, y_train,
        X_val_dummy, y_val,
        X_test_dummy, y_test,
        mean, std,
        x_train_raw, x_val_raw, x_test_raw
    ) = preprocess_mnist_multiclass(
        method="flatten",   # placeholder only; raw images are what matter here
        pca_components=75,  # placeholder only
        val_ratio=0.15
    )

    x_dev_raw = np.concatenate([x_train_raw, x_val_raw], axis=0)
    y_dev = np.concatenate([y_train, y_val], axis=0)

    print("Development set shape:", x_dev_raw.shape)
    print("Unique classes:", np.unique(y_dev))

    k_values = [1, 3, 5, 7]
    pca_components_list = [50, 75, 100]

    all_results = []
    best_result = None

    for pca_comp in pca_components_list:
        for k in k_values:
            print("\n" + "=" * 60)
            print(f"Testing configuration: PCA={pca_comp}, k={k}")
            print("=" * 60)

            cv_result = cross_validate_knn_pca(
                x_dev_raw=x_dev_raw,
                y_dev=y_dev,
                k_neighbors=k,
                pca_components=pca_comp,
                n_splits=5,
                random_state=42
            )

            row = {
                "pca_components": pca_comp,
                "k": k,
                **cv_result
            }
            all_results.append(row)

            print("\nCross-validation summary:")
            print(f"Accuracy    : {cv_result['accuracy_mean']:.4f} ± {cv_result['accuracy_std']:.4f}")
            print(f"Macro F1    : {cv_result['macro_f1_mean']:.4f} ± {cv_result['macro_f1_std']:.4f}")
            print(f"Weighted F1 : {cv_result['weighted_f1_mean']:.4f} ± {cv_result['weighted_f1_std']:.4f}")

        # Select best model based on ACCURACY
        if best_result is None:
         best_result = row
        else:
         if row["accuracy_mean"] > best_result["accuracy_mean"]:
          best_result = row
         elif row["accuracy_mean"] == best_result["accuracy_mean"]:
        # Tie-break 1: higher macro F1
           if row["macro_f1_mean"] > best_result["macro_f1_mean"]:
            best_result = row
           elif row["macro_f1_mean"] == best_result["macro_f1_mean"]:
            # Tie-break 2: smaller k
            if row["k"] < best_result["k"]:
                best_result = row
            # Tie-break 3: smaller PCA size
            elif row["k"] == best_result["k"] and row["pca_components"] < best_result["pca_components"]:
                best_result = row

    print("\n" + "=" * 75)
    print("ALL RESULTS SORTED BY MACRO F1")
    print("=" * 75)

    all_results_sorted = sorted(all_results, key=lambda r: r["macro_f1_mean"], reverse=True)

    for r in all_results_sorted:
        print(
            f"PCA={r['pca_components']:>3} | "
            f"k={r['k']:>2} | "
            f"Acc={r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f} | "
            f"MacroF1={r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f} | "
            f"WeightedF1={r['weighted_f1_mean']:.4f}±{r['weighted_f1_std']:.4f}"
        )

    return best_result, all_results

if __name__ == "__main__":
    best_result, all_results = tune_knn_with_5fold_cv()