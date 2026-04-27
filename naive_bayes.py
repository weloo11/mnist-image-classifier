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


def run_method(preprocessing, method, target_digit, pca_components):
    from gaussian_nb import GaussianNaiveBayes, evaluate

    X_train, y_train, X_val, y_val, X_test, y_test = preprocessing.preprocess_mnist(
        target_digit=target_digit,
        method=method,
        pca_components=pca_components,
        val_ratio=0.1
    )

    print(f"  Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    results = {}
    for split, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        y_pred = model.predict(X)
        results[split] = evaluate(y, y_pred)
        print_metrics(split, results[split])

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

    # Comparison table
    print(f"\n{'='*50}")
    print("  SUMMARY (Test Set)")
    print(f"{'='*50}")
    print(f"  {'Method':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*50}")
    for method, results in summary.items():
        m = results["Test"]
        print(f"  {method:<10} {m['accuracy']:>10.4f} {m['precision']:>10.4f}"
              f" {m['recall']:>10.4f} {m['f1']:>10.4f}")


if __name__ == "__main__":
    main()
