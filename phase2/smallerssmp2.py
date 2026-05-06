# ============================================================
# Phase 2 — Compact Multiclass Linear SVM from Scratch
# VGG16 CNN Features + One-vs-Rest SVM
# ============================================================

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.datasets import mnist
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input


# ============================================================
# 1. Settings
# ============================================================

N_CLASSES = 10
IMAGE_SIZE = 64
VAL_RATIO = 0.15
K_FOLDS = 3
OUTPUT_DIR = "phase2_svm_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)


# ============================================================
# 2. Load and split MNIST
# ============================================================

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    return x_train, y_train, x_test, y_test


def stratified_train_val_split(X, y, val_ratio=0.15, seed=42):
    np.random.seed(seed)

    train_idx = []
    val_idx = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)

        val_size = int(len(cls_idx) * val_ratio)

        val_idx.extend(cls_idx[:val_size])
        train_idx.extend(cls_idx[val_size:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ============================================================
# 3. VGG16 Feature Extraction
# ============================================================

def prepare_batch_for_vgg16(batch, image_size=64):
    """
    MNIST images are 28x28 grayscale.
    VGG16 needs image_size x image_size x 3 RGB images.
    """
    batch = batch[..., np.newaxis]
    batch = np.repeat(batch, 3, axis=-1)
    batch = tf.image.resize(batch, (image_size, image_size)).numpy()
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
    total = len(images)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        batch = images[start:end]
        batch_prepared = prepare_batch_for_vgg16(batch, IMAGE_SIZE)

        batch_features = extractor.predict(
            batch_prepared,
            batch_size=32,
            verbose=0
        )

        features.append(batch_features.astype(np.float32))

        print(f"Extracted CNN features: {end}/{total}", end="\r")

    print()

    del extractor
    gc.collect()

    return np.vstack(features).astype(np.float32)


def standardize_features(X_train, X_val, X_test):
    mean = np.mean(X_train, axis=0).astype(np.float32)
    std = np.std(X_train, axis=0).astype(np.float32)
    std[std == 0] = 1.0

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    return X_train, X_val, X_test


def standardize_train_val(X_train, X_val):
    mean = np.mean(X_train, axis=0).astype(np.float32)
    std = np.std(X_train, axis=0).astype(np.float32)
    std[std == 0] = 1.0

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)

    return X_train, X_val


# ============================================================
# 4. Metrics From Scratch
# ============================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    return cm


def precision_recall_f1(y_true, y_pred, n_classes=10):
    cm = confusion_matrix(y_true, y_pred, n_classes)

    precisions = []
    recalls = []
    f1s = []

    for cls in range(n_classes):
        tp = cm[cls, cls]
        fp = np.sum(cm[:, cls]) - tp
        fn = np.sum(cm[cls, :]) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return np.mean(precisions), np.mean(recalls), np.mean(f1s), cm


def per_class_metrics(y_true, y_pred, n_classes=10):
    cm = confusion_matrix(y_true, y_pred, n_classes)

    rows = []

    for cls in range(n_classes):
        tp = cm[cls, cls]
        fp = np.sum(cm[:, cls]) - tp
        fn = np.sum(cm[cls, :]) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        rows.append([cls, precision, recall, f1])

    return pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1-score"])


# ============================================================
# 5. Binary Linear SVM From Scratch
# ============================================================

class LinearSVMFromScratch:
    def __init__(
        self,
        learning_rate=0.01,
        lambda_param=0.0005,
        n_epochs=15,
        batch_size=512,
        lr_decay=0.95,
        patience=5
    ):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
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

    def compute_class_weights(self, y):
        n = len(y)

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == -1)

        w_pos = n / (2 * n_pos)
        w_neg = n / (2 * n_neg)

        return w_pos, w_neg

    def hinge_loss(self, X, y, w_pos, w_neg):
        scores = X @ self.w + self.b
        margins = y * scores

        sample_weights = np.where(y == 1, w_pos, w_neg)
        hinge = np.maximum(0, 1 - margins)

        loss = self.lambda_param * np.dot(self.w, self.w)
        loss += np.mean(sample_weights * hinge)

        return float(loss)

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

        w_pos, w_neg = self.compute_class_weights(y)

        lr = self.learning_rate
        no_improve = 0

        for epoch in range(self.n_epochs):
            idx = np.random.permutation(n_samples)

            X_shuffled = X[idx]
            y_shuffled = y[idx]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                scores = X_batch @ self.w + self.b
                margins = y_batch * scores

                sample_weights = np.where(y_batch == 1, w_pos, w_neg)
                active = margins < 1

                if np.any(active):
                    X_active = X_batch[active]
                    y_active = y_batch[active]
                    weights_active = sample_weights[active]

                    grad_w_hinge = -np.mean(
                        (weights_active * y_active)[:, None] * X_active,
                        axis=0
                    )

                    grad_b = -np.mean(weights_active * y_active)
                else:
                    grad_w_hinge = np.zeros_like(self.w)
                    grad_b = 0.0

                grad_w = 2 * self.lambda_param * self.w + grad_w_hinge

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


# ============================================================
# 6. Multiclass One-vs-Rest SVM
# ============================================================

class MulticlassOVRSVM:
    def __init__(
        self,
        n_classes=10,
        learning_rate=0.01,
        lambda_param=0.0005,
        n_epochs=15,
        batch_size=512,
        lr_decay=0.95,
        patience=5
    ):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.patience = patience

        self.models = []
        self.avg_train_losses = []
        self.avg_val_losses = []

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

            model = LinearSVMFromScratch(
                learning_rate=self.learning_rate,
                lambda_param=self.lambda_param,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                lr_decay=self.lr_decay,
                patience=self.patience
            )

            model.fit(X, y_binary, X_val, y_val_binary)

            self.models.append(model)
            train_curves.append(model.train_losses)
            val_curves.append(model.val_losses)

        self.avg_train_losses = self.average_curves(train_curves)
        self.avg_val_losses = self.average_curves(val_curves)

    def average_curves(self, curves):
        curves = [c for c in curves if len(c) > 0]

        if len(curves) == 0:
            return []

        min_len = min(len(c) for c in curves)
        curves = np.array([c[:min_len] for c in curves])

        return np.mean(curves, axis=0)

    def decision_function(self, X):
        scores = [model.decision_function(X) for model in self.models]
        return np.column_stack(scores)

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)


# ============================================================
# 7. Cross-Validation
# ============================================================

def stratified_kfold_indices(y, k=3, seed=42):
    np.random.seed(seed)

    folds = [[] for _ in range(k)]

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)

        split = np.array_split(idx, k)

        for i in range(k):
            folds[i].extend(split[i])

    return [np.array(fold) for fold in folds]


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
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_v, y_v = X[val_idx], y[val_idx]

            X_tr, X_v = standardize_train_val(X_tr, X_v)

            model = MulticlassOVRSVM(
                n_classes=N_CLASSES,
                learning_rate=config["learning_rate"],
                lambda_param=config["lambda_param"],
                n_epochs=config["n_epochs"],
                batch_size=config["batch_size"],
                lr_decay=config["lr_decay"],
                patience=config["patience"]
            )

            model.fit(X_tr, y_tr, X_v, y_v)

            y_pred = model.predict(X_v)
            _, _, f1, _ = precision_recall_f1(y_v, y_pred, N_CLASSES)

            fold_scores.append(f1)

            print(f"Fold {i + 1}/{k} | Macro F1 = {f1:.4f}")

        avg_f1 = np.mean(fold_scores)

        results.append({
            **config,
            "avg_macro_f1": avg_f1
        })

        if avg_f1 > best_score:
            best_score = avg_f1
            best_config = config

    return best_config, best_score, pd.DataFrame(results)


# ============================================================
# 8. Plotting Functions
# ============================================================

def plot_learning_curves(model):
    plt.figure(figsize=(8, 5))

    plt.plot(model.avg_train_losses, label="Train Loss")
    plt.plot(model.avg_val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves — One-vs-Rest Linear SVM")
    plt.legend()
    plt.grid(True)

    path = os.path.join(OUTPUT_DIR, "learning_curves.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


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
            color = "white" if cm[i, j] > threshold else "black"
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

    plt.bar(x - width, df["Precision"], width, label="Precision")
    plt.bar(x, df["Recall"], width, label="Recall")
    plt.bar(x + width, df["F1-score"], width, label="F1-score")

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


def save_sample_predictions(x_test_raw, y_true, y_pred, n_samples=25):
    np.random.seed(42)

    indices = np.random.choice(len(x_test_raw), n_samples, replace=False)

    grid_size = int(np.sqrt(n_samples))

    plt.figure(figsize=(10, 10))

    for i, idx in enumerate(indices):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(x_test_raw[idx], cmap="gray")

        true_label = y_true[idx]
        pred_label = y_pred[idx]

        title = f"T:{true_label} P:{pred_label}"
        plt.title(title, fontsize=9)
        plt.axis("off")

    plt.suptitle("Sample Test Predictions — True vs Predicted", fontsize=14)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "sample_predictions.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


# ============================================================
# 9. Bias-Variance / Regularization Analysis
# ============================================================

def bias_variance_analysis(X_train, y_train, X_val, y_val, base_config):
    lambda_values = [
        0.0,
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        50.0,
        100.0,
        500.0,
        1000.0
    ]

    print("\nRegularization / Bias-Variance Analysis")
    print("=" * 80)
    print(f"{'Lambda':>10} {'Train F1':>12} {'Val F1':>12} {'Gap':>10} {'Diagnosis':>15}")
    print("-" * 80)

    rows = []

    for lam in lambda_values:
        model = MulticlassOVRSVM(
            n_classes=N_CLASSES,
            learning_rate=base_config["learning_rate"],
            lambda_param=lam,
            n_epochs=base_config["n_epochs"],
            batch_size=base_config["batch_size"],
            lr_decay=base_config["lr_decay"],
            patience=base_config["patience"]
        )

        model.fit(X_train, y_train, X_val, y_val)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        _, _, train_f1, _ = precision_recall_f1(y_train, train_pred, N_CLASSES)
        _, _, val_f1, _ = precision_recall_f1(y_val, val_pred, N_CLASSES)

        gap = train_f1 - val_f1

        if train_f1 < 0.95 and val_f1 < 0.95:
            diagnosis = "Underfitting"
        elif gap > 0.05:
            diagnosis = "Overfitting"
        else:
            diagnosis = "Good fit"

        rows.append([lam, train_f1, val_f1, gap, diagnosis])

        print(
            f"{lam:>10.4f} "
            f"{train_f1:>12.4f} "
            f"{val_f1:>12.4f} "
            f"{gap:>10.4f} "
            f"{diagnosis:>15}"
        )

    print("-" * 80)

    df = pd.DataFrame(
        rows,
        columns=["lambda", "train_macro_f1", "val_macro_f1", "gap", "diagnosis"]
    )

    df.to_csv(
        os.path.join(OUTPUT_DIR, "bias_variance_analysis.csv"),
        index=False
    )

    plot_df = df[df["lambda"] > 0]

    plt.figure(figsize=(8, 5))
    plt.semilogx(
        plot_df["lambda"],
        plot_df["train_macro_f1"],
        "o-",
        label="Train Macro F1"
    )
    plt.semilogx(
        plot_df["lambda"],
        plot_df["val_macro_f1"],
        "o-",
        label="Validation Macro F1"
    )

    plt.xlabel("Lambda λ — log scale")
    plt.ylabel("Macro F1")
    plt.title("Regularization Effect — L2 Lambda vs Macro F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "regularization_effect.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", path)

    return df


# ============================================================
# 10. Main
# ============================================================

x_train_full, y_train_full, x_test, y_test = load_mnist()

x_train, y_train, x_val, y_val = stratified_train_val_split(
    x_train_full,
    y_train_full,
    VAL_RATIO
)

print("Train:", x_train.shape, y_train.shape)
print("Val  :", x_val.shape, y_val.shape)
print("Test :", x_test.shape, y_test.shape)


# ============================================================
# VGG16 Feature Extraction
# ============================================================

print("\nExtracting VGG16 features...")

X_train_cnn = extract_vgg16_features(x_train)
X_val_cnn = extract_vgg16_features(x_val)
X_test_cnn = extract_vgg16_features(x_test)

X_train, X_val, X_test = standardize_features(
    X_train_cnn,
    X_val_cnn,
    X_test_cnn
)

del X_train_cnn, X_val_cnn, X_test_cnn
gc.collect()


# ============================================================
# Cross-Validation
# ============================================================

param_grid = [
    {
        "learning_rate": 0.005,
        "lambda_param": 0.0001,
        "n_epochs": 15,
        "batch_size": 512,
        "lr_decay": 0.95,
        "patience": 5
    },
    {
        "learning_rate": 0.01,
        "lambda_param": 0.0005,
        "n_epochs": 15,
        "batch_size": 512,
        "lr_decay": 0.95,
        "patience": 5
    },
    {
        "learning_rate": 0.02,
        "lambda_param": 0.001,
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

cv_results.to_csv(
    os.path.join(OUTPUT_DIR, "cross_validation_results.csv"),
    index=False
)


# ============================================================
# Final Model Training
# ============================================================

print("\nTraining final model...")

final_model = MulticlassOVRSVM(
    n_classes=N_CLASSES,
    learning_rate=best_config["learning_rate"],
    lambda_param=best_config["lambda_param"],
    n_epochs=best_config["n_epochs"],
    batch_size=best_config["batch_size"],
    lr_decay=best_config["lr_decay"],
    patience=best_config["patience"]
)

final_model.fit(X_train, y_train, X_val, y_val)

plot_learning_curves(final_model)


# ============================================================
# Validation Evaluation
# ============================================================

y_val_pred = final_model.predict(X_val)

val_precision, val_recall, val_f1, val_cm = precision_recall_f1(
    y_val,
    y_val_pred,
    N_CLASSES
)

val_acc = accuracy(y_val, y_val_pred)

print("\nValidation Results")
print("=" * 40)
print(f"Accuracy       : {val_acc:.4f}")
print(f"Macro Precision: {val_precision:.4f}")
print(f"Macro Recall   : {val_recall:.4f}")
print(f"Macro F1       : {val_f1:.4f}")


# ============================================================
# Test Evaluation
# ============================================================

y_test_pred = final_model.predict(X_test)

test_precision, test_recall, test_f1, test_cm = precision_recall_f1(
    y_test,
    y_test_pred,
    N_CLASSES
)

test_acc = accuracy(y_test, y_test_pred)

print("\nFinal Test Results")
print("=" * 40)
print(f"Accuracy       : {test_acc:.4f}")
print(f"Macro Precision: {test_precision:.4f}")
print(f"Macro Recall   : {test_recall:.4f}")
print(f"Macro F1       : {test_f1:.4f}")

print("\nConfusion Matrix:")
print(test_cm)

pd.DataFrame(test_cm).to_csv(
    os.path.join(OUTPUT_DIR, "test_confusion_matrix.csv"),
    index=False
)

plot_confusion_matrix(
    test_cm,
    title="Test Confusion Matrix — VGG16 + One-vs-Rest SVM"
)


# ============================================================
# Per-Class Metrics
# ============================================================

per_class_df = per_class_metrics(y_test, y_test_pred, N_CLASSES)

per_class_df.to_csv(
    os.path.join(OUTPUT_DIR, "per_class_metrics.csv"),
    index=False
)

print("\nPer-class metrics:")
print(per_class_df.to_string(index=False))

plot_per_class_metrics(per_class_df)


# ============================================================
# Sample Prediction Photos
# ============================================================

save_sample_predictions(
    x_test_raw=x_test,
    y_true=y_test,
    y_pred=y_test_pred,
    n_samples=25
)


# ============================================================
# Bias-Variance / Regularization Analysis
# ============================================================

bias_variance_df = bias_variance_analysis(
    X_train,
    y_train,
    X_val,
    y_val,
    best_config
)


# ============================================================
# Summary
# ============================================================

summary = pd.DataFrame([
    {
        "Model": "Multiclass One-vs-Rest Linear SVM",
        "Feature Method": "VGG16 CNN Features",
        "Best Learning Rate": best_config["learning_rate"],
        "Best Lambda": best_config["lambda_param"],
        "Validation Accuracy": val_acc,
        "Validation Macro F1": val_f1,
        "Test Accuracy": test_acc,
        "Test Macro F1": test_f1
    }
])

summary.to_csv(
    os.path.join(OUTPUT_DIR, "summary.csv"),
    index=False
)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nDone.")
print("Outputs saved in:", OUTPUT_DIR)