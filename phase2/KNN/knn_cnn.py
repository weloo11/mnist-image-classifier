import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
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
    Simple multiclass KNN using Euclidean distance and majority voting.
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

        for i, x in enumerate(X):
            # Compute distance from x to all training samples
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # Get indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Get the labels of those nearest neighbors
            k_labels = self.y_train[k_indices]

            # Majority vote
            pred = np.bincount(k_labels).argmax()
            predictions.append(pred)

            if (i + 1) % 200 == 0:
                print(f"Predicted {i + 1}/{len(X)} samples...", flush=True)

        return np.array(predictions, dtype=int)


# ==========================================================
# 2) IMAGE PREPARATION FOR PRETRAINED CNN
# ==========================================================
def prepare_for_cnn(images, target_size=(96, 96)):
    """
    Converts MNIST grayscale images:
      shape: (N, 28, 28)
    into CNN-ready RGB images:
      shape: (N, H, W, 3)

    Steps:
      1) add channel dimension
      2) resize to CNN input size
      3) repeat grayscale channel into 3 channels
      4) apply MobileNetV2 preprocess_input
    """
    # Add channel dimension -> (N, 28, 28, 1)
    images = images[..., np.newaxis].astype(np.float32)

    # Resize to CNN input size
    images_resized = tf.image.resize(images, target_size).numpy()

    # Convert grayscale to RGB by repeating the single channel
    images_rgb = np.repeat(images_resized, 3, axis=-1)

    # Apply MobileNetV2 preprocessing
    images_rgb = preprocess_input(images_rgb)

    return images_rgb.astype(np.float32)


# ==========================================================
# 3) CNN FEATURE EXTRACTION
# ==========================================================
def extract_cnn_features(feature_extractor, images, batch_size=128):
    """
    Extracts deep features from images using the pretrained CNN.
    The CNN is used only as a fixed feature extractor.
    """
    features = feature_extractor.predict(images, batch_size=batch_size, verbose=1)
    return features.astype(np.float32)


# ==========================================================
# 4) EVALUATION HELPER
# ==========================================================
def evaluate_multiclass(y_true, y_pred):
    """
    Returns key multiclass metrics:
      - accuracy
      - macro F1
      - weighted F1
      - confusion matrix
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm
    }


# ==========================================================
# 5) LOAD MULTICLASS DATA
# ==========================================================
# IMPORTANT:
# We only need the RAW images here because the pretrained CNN
# will do its own feature extraction.
(
    X_train_dummy, y_train,
    X_val_dummy, y_val,
    X_test_dummy, y_test,
    mean, std,
    x_train_raw, x_val_raw, x_test_raw
) = preprocess_mnist_multiclass(
    method="flatten",        # placeholder only
    pca_components=75,   # placeholder only
    val_ratio=0.15
)

print("Raw train images:", x_train_raw.shape)
print("Raw val images  :", x_val_raw.shape)
print("Raw test images :", x_test_raw.shape)
print("Classes         :", np.unique(y_train))


# ==========================================================
# 6) PREPARE IMAGES FOR CNN
# ==========================================================
# MobileNetV2 needs larger RGB images, so we resize MNIST and
# convert grayscale to 3 channels.
print("\nPreparing train and validation images for CNN...", flush=True)

x_train_cnn = prepare_for_cnn(x_train_raw, target_size=(96, 96))
x_val_cnn = prepare_for_cnn(x_val_raw, target_size=(96, 96))

print("CNN-ready train shape:", x_train_cnn.shape)
print("CNN-ready val shape  :", x_val_cnn.shape)


# ==========================================================
# 7) LOAD PRETRAINED CNN AS FEATURE EXTRACTOR
# ==========================================================
# include_top=False removes the classifier head
# pooling='avg' gives one compact feature vector per image
print("\nLoading pretrained MobileNetV2...", flush=True)

feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(96, 96, 3)
)

print("Pretrained CNN loaded successfully.")


# ==========================================================
# 8) EXTRACT CNN FEATURES
# ==========================================================
print("\nExtracting CNN features for training set...", flush=True)
X_train_cnn_features = extract_cnn_features(feature_extractor, x_train_cnn, batch_size=128)

print("\nExtracting CNN features for validation set...", flush=True)
X_val_cnn_features = extract_cnn_features(feature_extractor, x_val_cnn, batch_size=128)

print("CNN feature train shape:", X_train_cnn_features.shape)
print("CNN feature val shape  :", X_val_cnn_features.shape)


# ==========================================================
# 9) OPTIONAL STANDARDIZATION OF CNN FEATURES
# ==========================================================
# KNN is distance-based, so standardizing the extracted features helps.
mean_cnn = np.mean(X_train_cnn_features, axis=0)
std_cnn = np.std(X_train_cnn_features, axis=0) + 1e-8

X_train_cnn_features = ((X_train_cnn_features - mean_cnn) / std_cnn).astype(np.float32)
X_val_cnn_features = ((X_val_cnn_features - mean_cnn) / std_cnn).astype(np.float32)


# ==========================================================
# 10) TUNE K ON VALIDATION SET
# ==========================================================
# Since the data are nearly balanced across the 10 classes,
# accuracy can be used as the primary selection criterion.
# Macro F1 is also reported as a secondary check.
k_values = [1, 3, 5, 7]

best_result = None
all_results = []

for k in k_values:
    print("\n" + "=" * 60)
    print(f"Testing KNN with pretrained CNN features, k = {k}")
    print("=" * 60)

    knn = KNNVectorized(k=k)
    knn.fit(X_train_cnn_features, y_train)

    y_val_pred = knn.predict(X_val_cnn_features)
    result = evaluate_multiclass(y_val, y_val_pred)

    all_results.append({
        "k": k,
        **result
    })

    print(f"Validation Accuracy    : {result['accuracy']:.4f}")
    print(f"Validation Macro F1    : {result['macro_f1']:.4f}")
    print(f"Validation Weighted F1 : {result['weighted_f1']:.4f}")

    if best_result is None:
        best_result = {
            "k": k,
            **result
        }
    else:
        # Primary selection: highest accuracy
        # Tie-breaker: higher macro F1
        # Final tie-breaker: smaller k
        if result["accuracy"] > best_result["accuracy"]:
            best_result = {
                "k": k,
                **result
            }
        elif result["accuracy"] == best_result["accuracy"]:
            if result["macro_f1"] > best_result["macro_f1"]:
                best_result = {
                    "k": k,
                    **result
                }
            elif result["macro_f1"] == best_result["macro_f1"] and k < best_result["k"]:
                best_result = {
                    "k": k,
                    **result
                }


# ==========================================================
# 11) PRINT FINAL VALIDATION COMPARISON
# ==========================================================
print("\n" + "=" * 70)
print("CNN FEATURE EXTRACTION + KNN VALIDATION SUMMARY")
print("=" * 70)

for r in all_results:
    print(
        f"k={r['k']:>2} | "
        f"Accuracy={r['accuracy']:.4f} | "
        f"Macro F1={r['macro_f1']:.4f} | "
        f"Weighted F1={r['weighted_f1']:.4f}"
    )

print("\nBest CNN+KNN configuration on validation set:")
print(f"k              : {best_result['k']}")
print(f"Accuracy       : {best_result['accuracy']:.4f}")
print(f"Macro F1       : {best_result['macro_f1']:.4f}")
print(f"Weighted F1    : {best_result['weighted_f1']:.4f}")

print("\nConfusion Matrix for best CNN+KNN validation result:")
print(best_result["confusion_matrix"])


# ==========================================================
# 12) OPTIONAL PER-CLASS REPORT ON VALIDATION
# ==========================================================
# This helps you understand which digits are hardest for CNN+KNN.
print("\nDetailed classification report for the best validation result:\n")

best_knn = KNNVectorized(k=best_result["k"])
best_knn.fit(X_train_cnn_features, y_train)
best_y_val_pred = best_knn.predict(X_val_cnn_features)

print(classification_report(y_val, best_y_val_pred, digits=4))


