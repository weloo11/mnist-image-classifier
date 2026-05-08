import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
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
        """
        KNN does not learn weights.
        It only stores the training data.
        """
        self.X_train = X.astype(np.float32)
        self.y_train = y.astype(int)

    def predict(self, X):
        """
        Predict each sample by:
        1) computing distance to all training samples
        2) selecting the k nearest neighbors
        3) taking the majority class among them
        """
        predictions = []

        for i, x in enumerate(X):
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]

            # Majority vote
            pred = np.bincount(k_labels).argmax()
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
    This function does the preprocessing INSIDE each CV fold.

    Why?
    Because PCA and scaling must be fit on fold-train only,
    then applied to fold-validation.
    Otherwise, we leak validation information into training.
    """

    # Step A: flatten images first
    X_train_flat = flatten_features(x_train_raw)
    X_val_flat = flatten_features(x_val_raw)

    # Step B: fit PCA only on fold-train
    pca = PCA(n_components=pca_components)
    X_train = pca.fit_transform(X_train_flat)
    X_val = pca.transform(X_val_flat)

    # Step C: standardize using fold-train statistics only
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)

    return X_train, X_val


# ==========================================================
# 4) ONE FULL 5-FOLD CV RUN FOR ONE PARAMETER COMBINATION
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
    Run 5-fold CV for ONE combination of:
      - k_neighbors
      - pca_components

    Returns the mean and std of:
      - accuracy
      - macro F1
      - weighted F1
    """

    # StratifiedKFold keeps class proportions balanced in each fold
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    fold_accuracies = []
    fold_macro_f1s = []
    fold_weighted_f1s = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_dev_raw, y_dev), start=1):
        # Split raw development images into fold-train and fold-validation
        x_fold_train = x_dev_raw[train_idx]
        y_fold_train = y_dev[train_idx]

        x_fold_val = x_dev_raw[val_idx]
        y_fold_val = y_dev[val_idx]

        # Apply PCA + scaling INSIDE this fold
        X_fold_train, X_fold_val = fit_transform_pca_fold(
            x_fold_train,
            x_fold_val,
            pca_components=pca_components
        )

        # Train KNN
        model = KNNVectorized(k=k_neighbors)
        model.fit(X_fold_train, y_fold_train)

        # Predict on fold-validation
        y_pred = model.predict(X_fold_val)

        # Compute evaluation metrics for this fold
        acc = accuracy_score(y_fold_val, y_pred)
        macro_f1 = f1_score(y_fold_val, y_pred, average="macro")
        weighted_f1 = f1_score(y_fold_val, y_pred, average="weighted")

        fold_accuracies.append(acc)
        fold_macro_f1s.append(macro_f1)
        fold_weighted_f1s.append(weighted_f1)

        print(
            f"Fold {fold_idx}: "
            f"Accuracy={acc:.4f}, "
            f"Macro F1={macro_f1:.4f}, "
            f"Weighted F1={weighted_f1:.4f}"
        )

    # Return the average CV performance
    return {
        "accuracy_mean": np.mean(fold_accuracies),
        "accuracy_std": np.std(fold_accuracies),
        "macro_f1_mean": np.mean(fold_macro_f1s),
        "macro_f1_std": np.std(fold_macro_f1s),
        "weighted_f1_mean": np.mean(fold_weighted_f1s),
        "weighted_f1_std": np.std(fold_weighted_f1s),
    }


# ==========================================================
# 5) MAIN TUNING FUNCTION
# ==========================================================
def tune_knn_with_5fold_cv():
    """
    Full tuning workflow:
    1) Load multiclass MNIST using your preprocessing file
    2) Combine train + validation into one development set
    3) Tune KNN over:
         - k
         - pca_components
    4) Select the best configuration using macro F1
    """

    # Load your multiclass data.
    # We only need the RAW images here because PCA must be fit fold-by-fold.
    (
        X_train_dummy, y_train,
        X_val_dummy, y_val,
        X_test_dummy, y_test,
        mean, std,
        x_train_raw, x_val_raw, x_test_raw
    ) = preprocess_mnist_multiclass(
        method="pca",       # placeholder; raw images are what matter here
        pca_components=75,  # placeholder
        val_ratio=0.15
    )

    # Combine train + validation into one development set
    # This is the data used for cross-validation.
    x_dev_raw = np.concatenate([x_train_raw, x_val_raw], axis=0)
    y_dev = np.concatenate([y_train, y_val], axis=0)

    print("Development set shape:", x_dev_raw.shape)
    print("Unique classes:", np.unique(y_dev))

    # Hyperparameter search space
    k_values = [1, 3, 5, 7]
    pca_components_list = [50, 75, 100]

    all_results = []
    best_result = None

    # Try all parameter combinations
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

            # Select best model based on macro F1
            if best_result is None:
                best_result = row
            else:
                if row["macro_f1_mean"] > best_result["macro_f1_mean"]:
                    best_result = row
                elif row["macro_f1_mean"] == best_result["macro_f1_mean"]:
                    # Tie-break 1: smaller k
                    if row["k"] < best_result["k"]:
                        best_result = row
                    # Tie-break 2: smaller PCA size
                    elif row["k"] == best_result["k"] and row["pca_components"] < best_result["pca_components"]:
                        best_result = row

    # Print all results sorted by macro F1
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

    # Print best configuration
    print("\n" + "=" * 75)
    print("BEST CONFIGURATION")
    print("=" * 75)
    print(f"PCA components : {best_result['pca_components']}")
    print(f"k              : {best_result['k']}")
    print(f"Accuracy       : {best_result['accuracy_mean']:.4f} ± {best_result['accuracy_std']:.4f}")
    print(f"Macro F1       : {best_result['macro_f1_mean']:.4f} ± {best_result['macro_f1_std']:.4f}")
    print(f"Weighted F1    : {best_result['weighted_f1_mean']:.4f} ± {best_result['weighted_f1_std']:.4f}")

    return best_result, all_results


# ==========================================================
# 6) RUN THE TUNING
# ==========================================================
if __name__ == "__main__":
    best_result, all_results = tune_knn_with_5fold_cv()