import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    return x_train, y_train, x_test, y_test


# ============================================================
# 2. SAFE PREPROCESSING (NO OOM)
# ============================================================

def preprocess_batch(batch):
    batch = np.expand_dims(batch, -1)
    batch = np.repeat(batch, 3, axis=-1)
    batch = tf.image.resize(batch, (224, 224))
    batch = preprocess_input(batch)
    return batch


# ============================================================
# 3. VGG16 FEATURE EXTRACTOR
# ============================================================

def build_vgg():
    model = VGG16(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    model.trainable = False
    return model


# ============================================================
# 4. MEMORY-SAFE FEATURE EXTRACTION
# ============================================================

def extract_features(model, X, batch_size=64):
    features = []
    n = len(X)

    for i in range(0, n, batch_size):
        batch = X[i:i+batch_size]

        batch = preprocess_batch(batch)

        feat = model.predict(batch, verbose=0)

        features.append(feat)

        print(f"Processed {min(i+batch_size, n)}/{n}")

    return np.concatenate(features, axis=0)


# ============================================================
# 5. SOFTMAX REGRESSION (FROM SCRATCH)
# ============================================================

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def one_hot(y, num_classes=10):
    Y = np.zeros((len(y), num_classes))
    Y[np.arange(len(y)), y] = 1
    return Y


def forward(X, W, b):
    return softmax(X @ W + b)


def loss_fn(A, Y, W, reg):
    N = Y.shape[0]
    ce = -np.sum(Y * np.log(A + 1e-8)) / N
    l2 = (reg / (2*N)) * np.sum(W * W)
    return ce + l2


def backward(X, A, Y, W, reg):
    N = X.shape[0]
    dZ = (A - Y) / N
    dW = X.T @ dZ + (reg / N) * W
    db = np.sum(dZ, axis=0, keepdims=True)
    return dW, db


def train(X, Y, X_val, Y_val, lr=0.01, reg=0.01, epochs=80):
    D = X.shape[1]
    C = Y.shape[1]

    W = np.random.randn(D, C) * 0.01
    b = np.zeros((1, C))

    for epoch in range(epochs):
        A = forward(X, W, b)
        loss = loss_fn(A, Y, W, reg)

        dW, db = backward(X, A, Y, W, reg)

        W -= lr * dW
        b -= lr * db

        if epoch % 10 == 0:
            val_pred = predict(X_val, W, b)
            val_acc = np.mean(val_pred == np.argmax(Y_val, axis=1))
            print(f"Epoch {epoch} | Loss {loss:.4f} | Val Acc {val_acc:.4f}")

    return W, b


def predict(X, W, b):
    return np.argmax(forward(X, W, b), axis=1)


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate(y_true, y_pred):
    print("\n================ FINAL RESULTS ================")

    print("\nAccuracy:", accuracy_score(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():
    x_train, y_train, x_test, y_test = load_data()

    # split train/val
    split = int(0.85 * len(x_train))

    x_tr, x_val = x_train[:split], x_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    Y_tr = one_hot(y_tr)
    Y_val = one_hot(y_val)

    print("\nLoading VGG16...")
    vgg = build_vgg()

    print("\nExtracting training features...")
    X_tr = extract_features(vgg, x_tr)

    print("\nExtracting validation features...")
    X_val = extract_features(vgg, x_val)

    print("\nExtracting test features...")
    X_test = extract_features(vgg, x_test)

    print("\nTraining softmax regression...")
    W, b = train(X_tr, Y_tr, X_val, Y_val, lr=0.01, reg=0.01, epochs=80)

    print("\nTesting...")
    preds = predict(X_test, W, b)

    evaluate(y_test, preds)


if __name__ == "__main__":
    main()


