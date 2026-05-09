# Phase 2 — Multiclass Linear SVM from Scratch
# Improvements:
# 1) VGG16 CNN Feature Extraction
# 2) K-Fold Cross-Validation
# 3) Overfitting / Underfitting Diagnosis using Plotted Learning Curves
# NOTE: No L1/L2 regularization is used in this code.

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.datasets import mnist
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input


N_CLASSES = 10
IMAGE_SIZE = 64
VAL_RATIO = 0.15
K_FOLDS = 3
OUTPUT_DIR = "phase2_svm_outputs"
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return (
        x_train.astype(np.float32),
        y_train.astype(np.int32),
        x_test.astype(np.float32),
        y_test.astype(np.int32)
    )


def stratified_split(X, y, val_ratio=0.15, seed=42):
    np.random.seed(seed)

    train_idx = []
    val_idx = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)

        n_val = int(len(idx) * val_ratio)

        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def prepare_vgg_batch(batch):
    batch = batch[..., np.newaxis]
    batch = np.repeat(batch, 3, axis=-1)
    batch = tf.image.resize(batch, (IMAGE_SIZE, IMAGE_SIZE)).numpy()
    batch = preprocess_input(batch)

    return batch


def extract_vgg16_features(images, batch_size=128):
    print("\nLoading VGG16 feature extractor...")

    extractor = VGG16(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    extractor.trainable = False

    features = []

    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))

        batch = prepare_vgg_batch(images[start:end])

        feats = extractor.predict(
            batch,
            batch_size=32,
            verbose=0
        )

        features.append(feats.astype(np.float32))

        print(f"Extracted CNN features: {end}/{len(images)}", end="\r")

    print()

    del extractor
    gc.collect()

    return np.vstack(features).astype(np.float32)


def standardize_train_val(X_train, X_val):
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)

    std[std == 0] = 1.0

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)

    return X_train, X_val


def standardize_all(X_train, X_val, X_test):
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)

    std[std == 0] = 1.0

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    return X_train, X_val, X_test


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1

    return cm


def metrics(y_true, y_pred, n_classes=10, as_dataframe=False):
    cm = confusion_matrix(y_true, y_pred, n_classes)

    rows = []

    for cls in range(n_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        rows.append([cls, precision, recall, f1])

    df = pd.DataFrame(
        rows,
        columns=["Class", "Precision", "Recall", "F1-score"]
    )

    if as_dataframe:
        return df, cm

    macro_precision = df["Precision"].mean()
    macro_recall = df["Recall"].mean()
    macro_f1 = df["F1-score"].mean()

    return macro_precision, macro_recall, macro_f1, cm


class LinearSVMFromScratch:
    def __init__(
        self,
        learning_rate=0.01,
        n_epochs=15,
        batch_size=512,
        lr_decay=0.95,
        patience=5
    ):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.patience = patience

        self.w = None
        self.b = 0.0

        self.train_losses = []
        self.val_losses = []

        self.best_w = None
        self.best_b = None
        self.best_val_loss = np.inf

    def class_weights(self, y):
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == -1)

        w_pos = len(y) / (2 * n_pos)
        w_neg = len(y) / (2 * n_neg)

        return w_pos, w_neg

    def hinge_loss(self, X, y, w_pos, w_neg):
        scores = X @ self.w + self.b
        margins = y * scores

        sample_weights = np.where(y == 1, w_pos, w_neg)
        hinge = np.maximum(0, 1 - margins)

        loss = np.mean(sample_weights * hinge)

        return float(loss)

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

        w_pos, w_neg = self.class_weights(y)

        lr = self.learning_rate
        no_improve = 0

        for epoch in range(self.n_epochs):
            idx = np.random.permutation(n_samples)

            X_s = X[idx]
            y_s = y[idx]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size

                X_b = X_s[start:end]
                y_b = y_s[start:end]

                scores = X_b @ self.w + self.b
                margins = y_b * scores

                sample_weights = np.where(y_b == 1, w_pos, w_neg)
                active = margins < 1

                if np.any(active):
                    X_active = X_b[active]
                    y_active = y_b[active]
                    weights_active = sample_weights[active]

                    grad_w = -np.mean(
                        (weights_active * y_active)[:, None] * X_active,
                        axis=0
                    )

                    grad_b = -np.mean(weights_active * y_active)

                else:
                    grad_w = np.zeros_like(self.w)
                    grad_b = 0.0

                self.w -= lr * grad_w
                self.b -= lr * grad_b

            train_loss = self.hinge_loss(X, y, w_pos, w_neg)
            self.train_losses.append(train_loss)

            if X_val is not None and y_val is not None:
                val_loss = self.hinge_loss(X_val, y_val, w_pos, w_neg)
                self.val_losses.append(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_w = self.w.copy()
                    self.best_b = self.b
                    no_improve = 0

                else:
                    no_improve += 1

                if no_improve >= self.patience:
                    break

            lr *= self.lr_decay

        if self.best_w is not None:
            self.w = self.best_w
            self.b = self.best_b

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class MulticlassOVRSVM:
    def __init__(
        self,
        n_classes=10,
        learning_rate=0.01,
        n_epochs=15,
        batch_size=512,
        lr_decay=0.95,
        patience=5
    ):
        self.n_classes = n_classes

        self.params = {
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr_decay": lr_decay,
            "patience": patience
        }

        self.models = []
        self.avg_train_losses = []
        self.avg_val_losses = []

    @staticmethod
    def average_curves(curves):
        curves = [curve for curve in curves if len(curve) > 0]

        if len(curves) == 0:
            return []

        min_len = min(len(curve) for curve in curves)
        curves = np.array([curve[:min_len] for curve in curves])

        return np.mean(curves, axis=0)

    def fit(self, X, y, X_val=None, y_val=None):
        self.models = []

        train_curves = []
        val_curves = []

        for cls in range(self.n_classes):
            print(f"Training class {cls} vs rest")

            y_binary = np.where(y == cls, 1, -1)

            if y_val is not None:
                y_val_binary = np.where(y_val == cls, 1, -1)
            else:
                y_val_binary = None

            model = LinearSVMFromScratch(**self.params)
            model.fit(X, y_binary, X_val, y_val_binary)

            self.models.append(model)

            train_curves.append(model.train_losses)
            val_curves.append(model.val_losses)

        self.avg_train_losses = self.average_curves(train_curves)
        self.avg_val_losses = self.average_curves(val_curves)

    def decision_function(self, X):
        scores = [model.decision_function(X) for model in self.models]
        return np.column_stack(scores)

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)


def stratified_kfold_indices(y, k=3, seed=42):
    np.random.seed(seed)

    folds = [[] for _ in range(k)]

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)

        split_classes = np.array_split(idx, k)

        for fold_index, split in enumerate(split_classes):
            folds[fold_index].extend(split)

    return [np.array(fold) for fold in folds]


def make_svm(config):
    return MulticlassOVRSVM(
        n_classes=N_CLASSES,
        learning_rate=config["learning_rate"],
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        lr_decay=config["lr_decay"],
        patience=config["patience"]
    )


def cross_validate_svm(X, y, param_grid, k=3):
    folds = stratified_kfold_indices(y, k=k)

    best_config = None
    best_score = -1
    results = []

    for config in param_grid:
        fold_scores = []

        print("\nTesting config:", config)

        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate(
                [folds[j] for j in range(k) if j != i]
            )

            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]

            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]

            X_train_fold, X_val_fold = standardize_train_val(
                X_train_fold,
                X_val_fold
            )

            model = make_svm(config)
            model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

            y_pred = model.predict(X_val_fold)

            _, _, fold_f1, _ = metrics(
                y_val_fold,
                y_pred,
                N_CLASSES
            )

            fold_scores.append(fold_f1)

            print(f"Fold {i + 1}/{k} | Macro F1 = {fold_f1:.4f}")

        avg_f1 = np.mean(fold_scores)

        results.append({
            **config,
            "avg_macro_f1": avg_f1
        })

        if avg_f1 > best_score:
            best_score = avg_f1
            best_config = config

    return best_config, best_score, pd.DataFrame(results)


def save_csv(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print("Saved:", path)


def plot_over_under_fitting(model):
    train_losses = np.array(model.avg_train_losses)
    val_losses = np.array(model.avg_val_losses)

    epochs = np.arange(1, len(train_losses) + 1)

    gap = val_losses - train_losses

    best_epoch = epochs[np.argmin(val_losses)]
    best_val_loss = np.min(val_losses)

    underfit_end = max(1, len(epochs) // 3)

    overfit_epochs = epochs[gap > 0.10]

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_losses,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        marker="o",
        label="Validation Loss"
    )

    plt.axvspan(
        epochs[0],
        epochs[underfit_end - 1],
        alpha=0.15,
        label="Underfitting Region"
    )

    if len(overfit_epochs) > 0:
        overfit_start = overfit_epochs[0]

        plt.axvspan(
            overfit_start,
            epochs[-1],
            alpha=0.15,
            label="Possible Overfitting Region"
        )

    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best Fit Epoch = {best_epoch}"
    )

    plt.scatter(
        best_epoch,
        best_val_loss,
        s=100,
        zorder=5
    )

    plt.text(
        best_epoch,
        best_val_loss,
        "  Best Fit",
        fontsize=10,
        verticalalignment="bottom"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Average Hinge Loss")
    plt.title("Overfitting and Underfitting Diagnosis Using Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "overfitting_underfitting_plot.png")

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


def plot_generalization_gap(model):
    train_losses = np.array(model.avg_train_losses)
    val_losses = np.array(model.avg_val_losses)

    epochs = np.arange(1, len(train_losses) + 1)

    gap = val_losses - train_losses

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        gap,
        marker="o",
        label="Validation Loss - Training Loss"
    )

    plt.axhline(
        y=0.10,
        linestyle="--",
        label="Overfitting Warning Threshold"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss Gap")
    plt.title("Generalization Gap Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "generalization_gap_curve.png")

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


def diagnose_from_graphs(model, train_pred, val_pred, y_train, y_val):
    _, _, train_f1, _ = metrics(y_train, train_pred, N_CLASSES)
    _, _, val_f1, _ = metrics(y_val, val_pred, N_CLASSES)

    train_losses = np.array(model.avg_train_losses)
    val_losses = np.array(model.avg_val_losses)

    f1_gap = train_f1 - val_f1
    loss_gap = val_losses[-1] - train_losses[-1]

    if train_f1 < 0.85 and val_f1 < 0.85:
        diagnosis = "Underfitting"

    elif f1_gap > 0.05 and loss_gap > 0.10:
        diagnosis = "Overfitting"

    else:
        diagnosis = "Good Fit"

    df = pd.DataFrame([
        {
            "train_macro_f1": train_f1,
            "validation_macro_f1": val_f1,
            "f1_gap": f1_gap,
            "final_train_loss": train_losses[-1],
            "final_validation_loss": val_losses[-1],
            "loss_gap": loss_gap,
            "diagnosis": diagnosis
        }
    ])

    save_csv(df, "overfitting_underfitting_diagnosis.csv")

    return df


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(9, 7))

    plt.imshow(cm)

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    labels = np.arange(cm.shape[0])

    plt.xticks(labels, labels)
    plt.yticks(labels, labels)

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > threshold:
                color = "white"
            else:
                color = "black"

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=color,
                fontsize=8
            )

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "test_confusion_matrix_heatmap.png")

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


def plot_per_class_metrics(df):
    x = np.arange(N_CLASSES)
    width = 0.25

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width,
        df["Precision"],
        width,
        label="Precision"
    )

    plt.bar(
        x,
        df["Recall"],
        width,
        label="Recall"
    )

    plt.bar(
        x + width,
        df["F1-score"],
        width,
        label="F1-score"
    )

    plt.xlabel("Digit Class")
    plt.ylabel("Score")
    plt.title("Per-Class Precision, Recall, and F1-score")

    plt.xticks(x)
    plt.ylim(0, 1.05)

    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "per_class_metrics.png")

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


def print_results(title, acc, precision, recall, f1):
    print(f"\n{title}")
    print("=" * 40)
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1        : {f1:.4f}")


def main():
    x_train_full, y_train_full, x_test, y_test = load_mnist_data()

    x_train, y_train, x_val, y_val = stratified_split(
        x_train_full,
        y_train_full,
        VAL_RATIO
    )

    print("Train:", x_train.shape, y_train.shape)
    print("Val  :", x_val.shape, y_val.shape)
    print("Test :", x_test.shape, y_test.shape)

    print("\nExtracting VGG16 features...")

    X_train_cnn = extract_vgg16_features(x_train)
    X_val_cnn = extract_vgg16_features(x_val)
    X_test_cnn = extract_vgg16_features(x_test)

    X_train, X_val, X_test = standardize_all(
        X_train_cnn,
        X_val_cnn,
        X_test_cnn
    )

    del X_train_cnn
    del X_val_cnn
    del X_test_cnn
    gc.collect()

    param_grid = [
        {
            "learning_rate": 0.005,
            "n_epochs": 15,
            "batch_size": 512,
            "lr_decay": 0.95,
            "patience": 5
        },
        {
            "learning_rate": 0.01,
            "n_epochs": 15,
            "batch_size": 512,
            "lr_decay": 0.95,
            "patience": 5
        },
        {
            "learning_rate": 0.02,
            "n_epochs": 15,
            "batch_size": 512,
            "lr_decay": 0.95,
            "patience": 5
        }
    ]

    best_config, best_cv_f1, cv_results = cross_validate_svm(
        X_train,
        y_train,
        param_grid,
        k=K_FOLDS
    )

    print("\nCross-validation results:")
    print(cv_results.to_string(index=False))

    print("\nBest config:", best_config)
    print("Best CV Macro F1:", best_cv_f1)

    save_csv(cv_results, "cross_validation_results.csv")

    print("\nTraining final model...")

    final_model = make_svm(best_config)

    final_model.fit(
        X_train,
        y_train,
        X_val,
        y_val
    )

    train_pred = final_model.predict(X_train)
    y_val_pred = final_model.predict(X_val)
    y_test_pred = final_model.predict(X_test)

    plot_over_under_fitting(final_model)
    plot_generalization_gap(final_model)

    diagnosis_df = diagnose_from_graphs(
        final_model,
        train_pred,
        y_val_pred,
        y_train,
        y_val
    )

    val_precision, val_recall, val_f1, _ = metrics(
        y_val,
        y_val_pred,
        N_CLASSES
    )

    val_acc = accuracy(y_val, y_val_pred)

    print_results(
        "Validation Results",
        val_acc,
        val_precision,
        val_recall,
        val_f1
    )

    test_precision, test_recall, test_f1, test_cm = metrics(
        y_test,
        y_test_pred,
        N_CLASSES
    )

    test_acc = accuracy(y_test, y_test_pred)

    print_results(
        "Final Test Results",
        test_acc,
        test_precision,
        test_recall,
        test_f1
    )

    print("\nConfusion Matrix:")
    print(test_cm)

    save_csv(
        pd.DataFrame(test_cm),
        "test_confusion_matrix.csv"
    )

    plot_confusion_matrix(
        test_cm,
        title="Test Confusion Matrix — VGG16 + One-vs-Rest Linear SVM"
    )

    per_class_df, _ = metrics(
        y_test,
        y_test_pred,
        N_CLASSES,
        as_dataframe=True
    )

    print("\nPer-class metrics:")
    print(per_class_df.to_string(index=False))

    save_csv(
        per_class_df,
        "per_class_metrics.csv"
    )

    plot_per_class_metrics(per_class_df)

    summary = pd.DataFrame([
        {
            "Model": "Multiclass One-vs-Rest Linear SVM",
            "Feature Method": "VGG16 CNN Features",
            "Improvement 1": "Pretrained CNN Feature Extraction",
            "Improvement 2": "K-Fold Cross-Validation",
            "Improvement 3": "Overfitting/Underfitting Diagnosis using Plotted Learning Curves",
            "Best Learning Rate": best_config["learning_rate"],
            "Diagnosis": diagnosis_df.loc[0, "diagnosis"],
            "Validation Accuracy": val_acc,
            "Validation Macro F1": val_f1,
            "Test Accuracy": test_acc,
            "Test Macro F1": test_f1
        }
    ])

    save_csv(
        summary,
        "summary.csv"
    )

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nDone.")
    print("Outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()