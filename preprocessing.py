import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from skimage.feature import hog


def preprocess_mnist(
    target_digit=5,
    method="flatten",
    pca_components=100,
    val_ratio=0.15
):
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

   
    y_train = np.where(y_train == target_digit, 1, 0)
    y_test = np.where(y_test == target_digit, 1, 0)

   
    x_train = x_train.astype(np.float64) / 255.0
    x_test = x_test.astype(np.float64) / 255.0

  
    np.random.seed(42)
    indices = np.random.permutation(len(x_train))

    x_train = x_train[indices]
    y_train = y_train[indices]

    val_size = int(len(x_train) * val_ratio)

    x_val = x_train[:val_size]
    y_val = y_train[:val_size]

    x_train = x_train[val_size:]
    y_train = y_train[val_size:]

    
    if method == "flatten":
        X_train = flatten_features(x_train)
        X_val = flatten_features(x_val)
        X_test = flatten_features(x_test)

    elif method == "pca":
        X_train_flat = flatten_features(x_train)
        X_val_flat = flatten_features(x_val)
        X_test_flat = flatten_features(x_test)

        pca = PCA(n_components=pca_components)

        X_train = pca.fit_transform(X_train_flat)
        X_val = pca.transform(X_val_flat)
        X_test = pca.transform(X_test_flat)

        print("PCA variance kept:", round(np.sum(pca.explained_variance_ratio_), 4))

    elif method == "hog":
        X_train = hog_features_dataset(x_train)
        X_val = hog_features_dataset(x_val)
        X_test = hog_features_dataset(x_test)

    else:
        raise ValueError("method must be 'flatten', 'pca', or 'hog'")

   
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_val, y_val, X_test, y_test, mean, std

def flatten_features(images):
    return images.reshape(images.shape[0], -1)


def hog_single_image(image):
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(7, 7),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )


def hog_features_dataset(images):
    features = []

    for i in range(images.shape[0]):
        features.append(hog_single_image(images[i]))

    return np.array(features)




if __name__ == "__main__":
    for method in ["flatten", "pca", "hog"]:
        print("\n==============================")
        print("Method:", method)
        print("==============================")

        X_train, y_train, X_val, y_val, X_test, y_test, mean, std = preprocess_mnist(
            target_digit=5,
            method=method,
            pca_components=100
        )

        print("X_train:", X_train.shape)
        print("X_val:", X_val.shape)
        print("X_test:", X_test.shape)
        print("y_train:", y_train.shape)
        