import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_mnist


# ==========================================================
# Settings
# ==========================================================

target_digit = 0
method = "hog"     # "flatten", "pca", or "hog"
pca_components = 100

learning_rate = 0.05
epochs = 300


# ==========================================================
# Load preprocessed data
# ==========================================================

X_train, y_train, X_val, y_val, X_test, y_test = preprocess_mnist(
    target_digit=target_digit,
    method=method,
    pca_components=pca_components
)

print("Preprocessing method:", method)
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)


# ==========================================================
# Logistic Regression Functions
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
    return np.where(probabilities >= 0.5, 1, 0)


def convert_to_class_1_class_2(predictions):
    return np.where(predictions == 1, 1, 2)


# ==========================================================
# Train Model
# ==========================================================

m, n = X_train.shape

W = np.zeros(n)
b = 0.0

train_losses = []
val_losses = []

for epoch in range(epochs):

    # Forward propagation
    y_pred = predict_probability(X_train, W, b)

    # Loss
    train_loss = compute_loss(y_train, y_pred)

    # Gradients
    error = y_pred - y_train

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
            "Epoch:", epoch,
            "| Train Loss:", round(train_loss, 4),
            "| Validation Loss:", round(val_loss, 4)
        )


# ==========================================================
# Evaluation
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


y_test_pred = predict_binary(X_test, W, b)

accuracy, precision, recall, f1, TP, TN, FP, FN = evaluate(y_test, y_test_pred)

final_predictions = convert_to_class_1_class_2(y_test_pred)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")
print("Target digit:", target_digit)
print("Class 1 = digit", target_digit)
print("Class 2 = all other digits")
print("Feature method:", method)

print("\nAccuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nConfusion Matrix:")
print("[[TN, FP],")
print(" [FN, TP]]")
print([[TN, FP],
       [FN, TP]])

print("\nFirst 20 predictions as Class 1 / Class 2:")
print(final_predictions[:20])


# ==========================================================
# Plot Loss Curve
# ==========================================================

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.title("Logistic Regression Learning Curve - " + method)
plt.legend()
plt.grid(True)
plt.show()