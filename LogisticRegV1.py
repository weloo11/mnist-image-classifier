import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist   # only loading dataset

# ==========================================================
# PHASE 1: LOGISTIC REGRESSION FROM SCRATCH
# Target Digit vs All Other Digits
# Target digit -> Class 1
# Other digits  -> Class 2
# ==========================================================

# -----------------------------
# 1. Load MNIST
# -----------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# -----------------------------
# 2. Choose target digit
# -----------------------------
target_digit = 0

# Class 1 = target digit
# Class 2 = all other digits

# For training logistic regression, we use:
# target digit -> 1
# other digits  -> 0
y_train_binary = np.where(y_train == target_digit, 1, 0)
y_test_binary = np.where(y_test == target_digit, 1, 0)

# -----------------------------
# 3. Preprocessing
# -----------------------------

# Flatten 28x28 images into 784 features
X_train = x_train.reshape(x_train.shape[0], 784).astype(np.float64)
X_test = x_test.reshape(x_test.shape[0], 784).astype(np.float64)

# Normalize pixels from [0,255] to [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Shuffle training data
np.random.seed(42)
indices = np.random.permutation(X_train.shape[0])

X_train = X_train[indices]
y_train_binary = y_train_binary[indices]

# Train / Validation split
val_ratio = 0.15
val_size = int(X_train.shape[0] * val_ratio)

X_val = X_train[:val_size]
y_val = y_train_binary[:val_size]

X_train = X_train[val_size:]
y_train_binary = y_train_binary[val_size:]

# Standardization
# Important: use training mean/std only
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

print("Training samples:", X_train.shape)
print("Validation samples:", X_val.shape)
print("Test samples:", X_test.shape)

print("\nClass distribution in training:")
print("Class 1 target digit:", np.sum(y_train_binary == 1))
print("Class 2 other digits:", np.sum(y_train_binary == 0))

# ==========================================================
# 4. Logistic Regression Manual Functions
# ==========================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_loss(y_true, y_pred):
    epsilon = 1e-8

    loss = -np.mean(
        y_true * np.log(y_pred + epsilon)
        +
        (1 - y_true) * np.log(1 - y_pred + epsilon)
    )

    return loss


def predict_probability(X, W, b):
    z = np.dot(X, W) + b
    return sigmoid(z)


def predict_binary(X, W, b):
    probabilities = predict_probability(X, W, b)
    predictions = np.where(probabilities >= 0.5, 1, 0)
    return predictions


def convert_to_class_1_class_2(predictions):
    return np.where(predictions == 1, 1, 2)

# ==========================================================
# 5. Train Logistic Regression Using Gradient Descent
# ==========================================================

m, n = X_train.shape

W = np.zeros(n)
b = 0.0

learning_rate = 0.05
epochs = 300

train_losses = []
val_losses = []

for epoch in range(epochs):

    # Forward propagation
    y_pred = predict_probability(X_train, W, b)

    # Loss
    train_loss = comput_loss(y_train_binary, y_pred)

    # Error
    error = y_pred - y_train_binary

    # Gradients
    dW = (1 / m) * np.dot(X_train.T, error)
    db = (1 / m) * np.sum(error)

    # Update parameters
    W = W - learning_rate * dW
    b = b - learning_rate * db

    # Validation loss
    val_pred = predict_probability(X_val, W, b)
    val_loss = compute_loss(y_val, val_pred)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 20 == 0:
        print(
            "Epoch:",
            epoch,
            "| Train Loss:",
            round(train_loss, 4),
            "| Validation Loss:",
            round(val_loss, 4)
        )

# ==========================================================
# 6. Evaluation Metrics
# ==========================================================

def evaluate(y_true, y_pred):

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    return accuracy, precision, recall, f1_score, TP, TN, FP, FN


# Predict test data
y_test_pred_binary = predict_binary(X_test, W, b)

accuracy, precision, recall, f1, TP, TN, FP, FN = evaluate(
    y_test_binary,
    y_test_pred_binary
)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")
print("Target digit:", target_digit)
print("Class 1 = digit", target_digit)
print("Class 2 = all other digits")

print("\nAccuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nConfusion Matrix:")
print("Rows = Actual, Columns = Predicted")
print("[[TN, FP],")
print(" [FN, TP]]")
print([[TN, FP],
       [FN, TP]])

# Convert prediction output to Class 1 / Class 2
final_class_predictions = convert_to_class_1_class_2(y_test_pred_binary)

print("\nExample final class predictions:")
print(final_class_predictions[:20])

# ==========================================================
# 7. Plot Loss Curve
# ==========================================================

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.title("Logistic Regression Learning Curve")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================================
# 8. Show Sample Predictions
# ==========================================================

plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    # Undo standardization for display
    image = X_test[i] * std + mean
    image = image.reshape(28, 28)

    predicted_class = final_class_predictions[i]

    actual_class = 1 if y_test_binary[i] == 1 else 2

    plt.imshow(image, cmap="gray")
    plt.title(
        "Actual C" + str(actual_class) +
        "\nPred C" + str(predicted_class)
    )
    plt.axis("off")

plt.tight_layout()
plt.show()