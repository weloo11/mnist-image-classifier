import contextlib
import importlib
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1'))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt





def multiclass_evaluate(y_true, y_pred, n_classes=10):
    acc = np.mean(y_true == y_pred)

    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t][p] += 1

    per_class = {}
    precisions, recalls, f1s = [], [], []

    for c in range(n_classes):
        tp = conf_matrix[c, c]
        fp = int(conf_matrix[:, c].sum()) - tp
        fn = int(conf_matrix[c, :].sum()) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy":       acc,
        "precision":      np.mean(precisions),
        "recall":         np.mean(recalls),
        "f1":             np.mean(f1s),
        "per_class":      per_class,
        "confusion_matrix": conf_matrix,
    }


def print_confusion_matrix(conf_matrix):
    n = conf_matrix.shape[0]
    print("         " + "".join(f"  P{c}" for c in range(n)))
    for r in range(n):
        row = f"  True {r}  " + "".join(f"{conf_matrix[r, c]:5d}" for c in range(n))
        print(row)


def print_metrics(split, metrics, show_conf_matrix=False, show_per_class=False):
    m = metrics
    print(f"  {split}: acc={m['accuracy']:.4f}  prec={m['precision']:.4f}"
          f"  rec={m['recall']:.4f}  f1={m['f1']:.4f}")

    if show_per_class:
        print(f"  {'Digit':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*42}")
        for c, cm in m["per_class"].items():
            print(f"  {c:<8} {cm['precision']:>10.4f} {cm['recall']:>10.4f} {cm['f1']:>10.4f}")

    if show_conf_matrix:
        print("  Confusion Matrix (rows=True label, cols=Predicted):")
        print_confusion_matrix(m["confusion_matrix"])


def extract_cnn_features(x_raw, batch_size=256):
    """
    Extract features from raw 28x28 MNIST images using pretrained MobileNetV2.
    - Resizes to 96x96 (MobileNetV2 minimum) and converts grayscale to RGB.
    - Returns 1280-dim global-average-pooled features per image.
    MobileNetV2 is chosen because it is lightweight, fast, and already available
    through tensorflow.keras which the project uses for MNIST loading.
    """
    import tensorflow as tf

    MobileNetV2     = tf.keras.applications.MobileNetV2
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    print(f"    Resizing {x_raw.shape[0]} images to 96x96 RGB ...")
    x_4d = x_raw[..., np.newaxis]                          # (n,28,28,1)
    x_resized = tf.image.resize(x_4d, [96, 96]).numpy()   # (n,96,96,1)
    x_rgb = np.repeat(x_resized, 3, axis=-1)               # (n,96,96,3)
    # preprocess_input expects [0,255] and scales internally to [-1,1]
    x_rgb = preprocess_input(x_rgb * 255.0)

    print("    Loading pretrained MobileNetV2 (ImageNet weights) ...")
    base_model = MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        pooling="avg",
        weights="imagenet",
    )
    base_model.trainable = False

    print("    Extracting features ...")
    features = base_model.predict(x_rgb, batch_size=batch_size, verbose=0)
    print(f"    CNN features shape: {features.shape}")
    return features   # (n, 1280)



def kfold_cross_validate(X, y, k=5, var_smoothing=1e-8):
    from gaussian_nb import GaussianNaiveBayes

    n = len(y)
    indices = np.arange(n)
    fold_size = n // k
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    fold_metrics = {key: [] for key in metric_keys}

    for i in range(k):
        val_idx   = indices[i * fold_size: (i + 1) * fold_size]
        train_idx = np.concatenate([
            indices[:i * fold_size],
            indices[(i + 1) * fold_size:]
        ])

        model = GaussianNaiveBayes(var_smoothing=var_smoothing)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        m = multiclass_evaluate(y[val_idx], y_pred)

        for key in metric_keys:
            fold_metrics[key].append(m[key])

    return {
        key: (np.mean(fold_metrics[key]), np.std(fold_metrics[key]))
        for key in metric_keys
    }


def plot_learning_curves(X_train, y_train, X_val, y_val, var_smoothing, method, output_dir="."):
    from gaussian_nb import GaussianNaiveBayes

    fractions = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    n = len(y_train)
    train_accs, val_accs = [], []

    for frac in fractions:
        size = max(10, int(n * frac))
        X_sub, y_sub = X_train[:size], y_train[:size]

        model = GaussianNaiveBayes(var_smoothing=var_smoothing)
        model.fit(X_sub, y_sub)

        train_accs.append(multiclass_evaluate(y_sub, model.predict(X_sub))["accuracy"])
        val_accs.append(  multiclass_evaluate(y_val, model.predict(X_val))["accuracy"])

    gap = train_accs[-1] - val_accs[-1]
    if val_accs[-1] < 0.60:
        diagnosis = "Underfitting — both curves low (high bias)"
    elif gap > 0.10:
        diagnosis = "Overfitting — large train/val gap (high variance)"
    else:
        diagnosis = "Good fit — train and val curves are close"

    pct_labels = [int(f * 100) for f in fractions]
    plt.figure(figsize=(8, 5))
    plt.plot(pct_labels, train_accs, marker="o", label="Train Accuracy")
    plt.plot(pct_labels, val_accs,   marker="s", label="Val Accuracy")
    plt.xlabel("% of Training Data")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curves — {method.upper()}\n{diagnosis}")
    plt.xticks(pct_labels)
    plt.legend()
    plt.tight_layout()
    fname = f"{output_dir}/nb2_learning_curve_{method}.png"
    plt.savefig(fname)
    plt.close()
    print(f"  Learning curve saved -> {fname}")
    print(f"  Diagnosis: {diagnosis}  (train={train_accs[-1]:.4f}, val={val_accs[-1]:.4f})")
    return {"train_acc": train_accs, "val_acc": val_accs, "fractions": fractions, "diagnosis": diagnosis}


def run_method(preprocessing, method, pca_components, k_folds=5, output_dir="."):
    from gaussian_nb import GaussianNaiveBayes
    from sklearn.decomposition import PCA

    if method == "cnn":
        # Load split via flatten (cheap) — we only need raw images and labels
        result = preprocessing.preprocess_mnist_multiclass(
            method="flatten", pca_components=pca_components, val_ratio=0.15
        )
        _, y_train, _, y_val, _, y_test = result[:6]
        x_train_raw, x_val_raw, x_test_raw = result[8], result[9], result[10]

        print("  Extracting MobileNetV2 features (train) ...")
        X_train = extract_cnn_features(x_train_raw)
        print("  Extracting MobileNetV2 features (val) ...")
        X_val   = extract_cnn_features(x_val_raw)
        print("  Extracting MobileNetV2 features (test) ...")
        X_test  = extract_cnn_features(x_test_raw)

        # PCA to decorrelate CNN features — GNB assumes feature independence
        print(f"  Applying PCA({pca_components}) to decorrelate CNN features ...")
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_val   = pca.transform(X_val)
        X_test  = pca.transform(X_test)
        var_kept = np.sum(pca.explained_variance_ratio_)
        print(f"  PCA variance retained: {var_kept:.4f}")

        # Standardize
        mean = np.mean(X_train, axis=0)
        std  = np.std(X_train,  axis=0) + 1e-8
        X_train = ((X_train - mean) / std).astype(np.float32)
        X_val   = ((X_val   - mean) / std).astype(np.float32)
        X_test  = ((X_test  - mean) / std).astype(np.float32)
    else:
        result = preprocessing.preprocess_mnist_multiclass(
            method=method, pca_components=pca_components, val_ratio=0.15
        )
        X_train, y_train, X_val, y_val, X_test, y_test = result[:6]

    print(f"  Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    VAR_SMOOTHING = 1e-9

    X_cv = np.concatenate([X_train, X_val], axis=0)
    y_cv = np.concatenate([y_train, y_val], axis=0)

    cv_results = kfold_cross_validate(X_cv, y_cv, k=k_folds, var_smoothing=VAR_SMOOTHING)
    print(f"\n  {k_folds}-Fold Cross-Validation (mean ± std):")
    for key, (mean, std) in cv_results.items():
        print(f"    {key:<10}: {mean:.4f} ± {std:.4f}")

    model = GaussianNaiveBayes(var_smoothing=VAR_SMOOTHING)
    model.fit(X_cv, y_cv)

    print("\n  Final evaluation:")
    results = {}
    for split, X, y in [("Train+Val", X_cv, y_cv), ("Test", X_test, y_test)]:
        y_pred = model.predict(X)
        results[split] = multiclass_evaluate(y, y_pred)
        is_test = (split == "Test")
        print_metrics(split, results[split],
                      show_conf_matrix=is_test,
                      show_per_class=is_test)

    results["CV"] = cv_results

    print("\n  --- Learning Curves ---")
    results["learning_curves"] = plot_learning_curves(
        X_train, y_train, X_val, y_val, VAR_SMOOTHING, method, output_dir
    )

    return results


def main():
    PCA_COMPONENTS = 100
    METHODS        = ["flatten", "pca", "hog", "cnn"]
    K_FOLDS        = 5
    OUTPUT_DIR     = "phase2_nb_outputs"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import importlib.util, pathlib
    _spec = importlib.util.spec_from_file_location(
        "preprocessing2",
        pathlib.Path(__file__).parent / "preprocessing2.py"
    )
    preprocessing = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(preprocessing)

    summary = {}
    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"  Method: {method.upper()}")
        print(f"{'='*60}")
        summary[method] = run_method(preprocessing, method, PCA_COMPONENTS, K_FOLDS, OUTPUT_DIR)

    print(f"\n{'='*65}")
    print("  SUMMARY — Phase 2 Multiclass Gaussian Naive Bayes (10 Classes)")
    print(f"{'='*65}")
    print(f"  {'Method':<10} {'CV Acc (mean±std)':>22} {'Test Acc':>10} {'Test F1':>10}")
    print(f"  {'-'*55}")
    for method, results in summary.items():
        test            = results["Test"]
        cv_mean, cv_std = results["CV"]["accuracy"]
        print(f"  {method:<10}"
              f" {f'{cv_mean:.4f}±{cv_std:.4f}':>22}"
              f" {test['accuracy']:>10.4f}"
              f" {test['f1']:>10.4f}")


if __name__ == "__main__":
    main()
