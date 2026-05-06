import contextlib
import importlib
import io
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_preprocessing():
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import preprocessing2
        return preprocessing2
    except Exception as e:
        print("preprocessing2 import failed:", e)
        print("Installing required packages: tensorflow, scikit-image, scikit-learn")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "tensorflow", "scikit-image", "scikit-learn"
        ])
        importlib.invalidate_caches()
        with contextlib.redirect_stdout(io.StringIO()):
            return importlib.import_module("preprocessing2")


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


def tune_var_smoothing(X_train, y_train, X_val, y_val,
                       candidates=None):
    from gaussian_nb import GaussianNaiveBayes

    if candidates is None:
        candidates = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]

    best_f1 = -1.0
    best_vs = candidates[0]

    print("  Hyperparameter tuning (var_smoothing on validation set):")
    for vs in candidates:
        model = GaussianNaiveBayes(var_smoothing=vs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        m = multiclass_evaluate(y_val, y_pred)
        print(f"    var_smoothing={vs:.0e}  ->  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_vs = vs

    print(f"  Best var_smoothing: {best_vs:.0e}  (val F1={best_f1:.4f})")
    return best_vs


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
    """Train on increasing fractions of training data; plot train vs val F1."""
    from gaussian_nb import GaussianNaiveBayes

    fractions = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    n = len(y_train)
    train_f1s, val_f1s, sizes = [], [], []

    for frac in fractions:
        size = max(10, int(n * frac))
        X_sub, y_sub = X_train[:size], y_train[:size]

        model = GaussianNaiveBayes(var_smoothing=var_smoothing)
        model.fit(X_sub, y_sub)

        train_f1s.append(multiclass_evaluate(y_sub,   model.predict(X_sub))["f1"])
        val_f1s.append(  multiclass_evaluate(y_val,   model.predict(X_val))["f1"])
        sizes.append(size)

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_f1s, marker="o", label="Train F1")
    plt.plot(sizes, val_f1s,   marker="s", label="Val F1")
    plt.xlabel("Training set size")
    plt.ylabel("Macro F1")
    plt.title(f"Learning Curves — {method.upper()} (var_smoothing={var_smoothing:.0e})")
    plt.legend()
    plt.tight_layout()
    fname = f"{output_dir}/nb2_learning_curve_{method}.png"
    plt.savefig(fname)
    plt.close()
    print(f"  Learning curve saved -> {fname}")

    gap = train_f1s[-1] - val_f1s[-1]
    if val_f1s[-1] < 0.60:
        diagnosis = "Underfitting (high bias) — both curves low"
    elif gap > 0.10:
        diagnosis = "Overfitting (high variance) — large train/val gap"
    else:
        diagnosis = "Good fit — train and val curves close"
    print(f"  Diagnosis: {diagnosis}  (train F1={train_f1s[-1]:.4f}, val F1={val_f1s[-1]:.4f})")
    return {"train_f1": train_f1s, "val_f1": val_f1s, "sizes": sizes, "diagnosis": diagnosis}


def bias_variance_analysis(X_train, y_train, X_val, y_val, method, output_dir="."):
    """Sweep var_smoothing over a wide range; plot train vs val F1 to show bias-variance tradeoff."""
    from gaussian_nb import GaussianNaiveBayes

    smoothing_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    train_f1s, val_f1s = [], []

    print("  Bias-Variance Analysis (sweeping var_smoothing):")
    for vs in smoothing_values:
        model = GaussianNaiveBayes(var_smoothing=vs)
        model.fit(X_train, y_train)
        tr_f1  = multiclass_evaluate(y_train, model.predict(X_train))["f1"]
        val_f1 = multiclass_evaluate(y_val,   model.predict(X_val))["f1"]
        train_f1s.append(tr_f1)
        val_f1s.append(val_f1)
        print(f"    vs={vs:.0e}  train_f1={tr_f1:.4f}  val_f1={val_f1:.4f}")

    log_vs = [np.log10(vs) for vs in smoothing_values]
    plt.figure(figsize=(8, 5))
    plt.plot(log_vs, train_f1s, marker="o", label="Train F1")
    plt.plot(log_vs, val_f1s,   marker="s", label="Val F1")
    plt.xlabel("log10(var_smoothing)")
    plt.ylabel("Macro F1")
    plt.title(f"Bias-Variance Analysis — {method.upper()}")
    plt.legend()
    plt.tight_layout()
    fname = f"{output_dir}/nb2_bias_variance_{method}.png"
    plt.savefig(fname)
    plt.close()
    print(f"  Bias-variance plot saved -> {fname}")
    return {"smoothing_values": smoothing_values, "train_f1": train_f1s, "val_f1": val_f1s}


def run_method(preprocessing, method, pca_components, k_folds=5, output_dir="."):
    from gaussian_nb import GaussianNaiveBayes

    result = preprocessing.preprocess_mnist_multiclass(
        method=method,
        pca_components=pca_components,
        val_ratio=0.15
    )
    X_train, y_train, X_val, y_val, X_test, y_test = result[:6]

    print(f"  Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    best_vs = tune_var_smoothing(X_train, y_train, X_val, y_val)

    X_cv = np.concatenate([X_train, X_val], axis=0)
    y_cv = np.concatenate([y_train, y_val], axis=0)

    cv_results = kfold_cross_validate(X_cv, y_cv, k=k_folds, var_smoothing=best_vs)
    print(f"\n  {k_folds}-Fold Cross-Validation (mean ± std):")
    for key, (mean, std) in cv_results.items():
        print(f"    {key:<10}: {mean:.4f} ± {std:.4f}")

    model = GaussianNaiveBayes(var_smoothing=best_vs)
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

    results["CV"]      = cv_results
    results["best_vs"] = best_vs

    print("\n  --- Learning Curves ---")
    results["learning_curves"] = plot_learning_curves(
        X_train, y_train, X_val, y_val, best_vs, method, output_dir
    )

    print("\n  --- Bias-Variance Analysis ---")
    results["bias_variance"] = bias_variance_analysis(
        X_train, y_train, X_val, y_val, method, output_dir
    )

    return results


def main():
    PCA_COMPONENTS = 100
    METHODS        = ["flatten", "pca", "hog"]
    K_FOLDS        = 5
    OUTPUT_DIR     = "phase2_nb_outputs"

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    preprocessing = ensure_preprocessing()

    summary = {}
    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"  Method: {method.upper()}")
        print(f"{'='*60}")
        summary[method] = run_method(preprocessing, method, PCA_COMPONENTS, K_FOLDS, OUTPUT_DIR)

    print(f"\n{'='*72}")
    print("  SUMMARY — Phase 2 Multiclass Gaussian Naive Bayes (10 Classes)")
    print(f"{'='*72}")
    print(f"  {'Method':<10} {'Best VS':>10} {'CV Acc (mean±std)':>22} {'Test Acc':>10} {'Test F1':>10}")
    print(f"  {'-'*65}")
    for method, results in summary.items():
        test            = results["Test"]
        cv_mean, cv_std = results["CV"]["accuracy"]
        best_vs         = results["best_vs"]
        print(f"  {method:<10} {best_vs:>10.0e}"
              f" {f'{cv_mean:.4f}±{cv_std:.4f}':>22}"
              f" {test['accuracy']:>10.4f}"
              f" {test['f1']:>10.4f}")


if __name__ == "__main__":
    main()
