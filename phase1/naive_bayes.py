import contextlib
import importlib
import io
import subprocess
import sys
import numpy as np


def ensure_preprocessing():
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import preprocessing
        return preprocessing
    except Exception as e:
        print("preprocessing import failed:", e)
        print("Installing required packages: tensorflow, scikit-image, scikit-learn")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "tensorflow", "scikit-image", "scikit-learn"
        ])
        importlib.invalidate_caches()
        with contextlib.redirect_stdout(io.StringIO()):
            return importlib.import_module("preprocessing")


def print_confusion_matrix(tp, tn, fp, fn):
    print("  Confusion Matrix:")
    print(f"    {'':10s} Pred 0   Pred 1")
    print(f"    {'True 0':10s} {tn:<8} {fp:<8}")
    print(f"    {'True 1':10s} {fn:<8} {tp:<8}")


def print_metrics(split, metrics):
    m = metrics
    print(f"  {split}: acc={m['accuracy']:.4f}  prec={m['precision']:.4f}"
          f"  rec={m['recall']:.4f}  f1={m['f1']:.4f}")
    if split == "Test":
        print_confusion_matrix(m['tp'], m['tn'], m['fp'], m['fn'])


def kfold_cross_validate(X, y, k=5):
    from gaussian_nb import GaussianNaiveBayes, evaluate

    n = len(y)
    indices = np.arange(n)
    fold_size = n // k
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    fold_metrics = {key: [] for key in metric_keys}

    for i in range(k):
        val_idx = indices[i * fold_size: (i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        model = GaussianNaiveBayes()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        m = evaluate(y[val_idx], y_pred)

        for key in metric_keys:
            fold_metrics[key].append(m[key])

    return {key: (np.mean(fold_metrics[key]), np.std(fold_metrics[key])) for key in metric_keys}


def run_method(preprocessing, method, target_digit, pca_components, k_folds=5):
    from gaussian_nb import GaussianNaiveBayes, evaluate

    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = preprocessing.preprocess_mnist(
        target_digit=target_digit,
        method=method,
        pca_components=pca_components,
        val_ratio=0.15
    )

    print(f"  Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    X_cv = np.concatenate([X_train, X_val], axis=0)
    y_cv = np.concatenate([y_train, y_val], axis=0)
    cv_results = kfold_cross_validate(X_cv, y_cv, k=k_folds)
    print(f"\n  {k_folds}-Fold Cross-Validation (mean ± std):")
    for key, (mean, std) in cv_results.items():
        print(f"    {key:<10}: {mean:.4f} ± {std:.4f}")

    model = GaussianNaiveBayes()
    model.fit(X_cv, y_cv)

    print("\n  Final evaluation:")
    results = {}
    for split, X, y in [("Train+Val", X_cv, y_cv), ("Test", X_test, y_test)]:
        y_pred = model.predict(X)
        results[split] = evaluate(y, y_pred)
        print_metrics(split, results[split])

    results["CV"] = cv_results
    return results


def main():
    TARGET_DIGIT = 5
    PCA_COMPONENTS = 100
    METHODS = ["flatten", "pca", "hog"]

    preprocessing = ensure_preprocessing()

    summary = {}
    for method in METHODS:
        print(f"\n{'='*50}")
        print(f"  Method: {method.upper()}")
        print(f"{'='*50}")
        summary[method] = run_method(preprocessing, method, TARGET_DIGIT, PCA_COMPONENTS)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<10} {'CV Acc (mean±std)':>20} {'Test Acc':>10} {'Test F1':>10}")
    print(f"  {'-'*55}")
    for method, results in summary.items():
        test = results["Test"]
        cv_mean, cv_std = results["CV"]["accuracy"]
        print(f"  {method:<10} {f'{cv_mean:.4f}±{cv_std:.4f}':>20}"
              f" {test['accuracy']:>10.4f} {test['f1']:>10.4f}")


if __name__ == "__main__":
    main()
