import numpy as np
from preprocessing import preprocess_mnist


class KNNLoop:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict_one(self, x):
        distances = []

        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda pair: pair[0])
        k_nearest_labels = [label for _, label in distances[:self.k]]

        prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return prediction

    def predict(self, X):
        predictions = []
        for i, x in enumerate(X):
            pred = self.predict_one(x)
            predictions.append(pred)

            if (i + 1) % 50 == 0:
                print(f"  Predicted {i + 1}/{len(X)} samples...", flush=True)

        return np.array(predictions)


def confusion_matrix_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


def accuracy_score_manual(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return 0.0 if (tp + fp) == 0 else tp / (tp + fp)


def recall_score_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return 0.0 if (tp + fn) == 0 else tp / (tp + fn)


def f1_score_manual(y_true, y_pred):
    p = precision_score_manual(y_true, y_pred)
    r = recall_score_manual(y_true, y_pred)
    return 0.0 if (p + r) == 0 else 2 * (p * r) / (p + r)


def main():
    target_digit = 5
    val_subset_size = 500
    pca_components_list = [ 100]
    k_values = [3]

    best_overall = None
    all_results = []

    for pca_components in pca_components_list:
        print("\n" + "=" * 60)
        print(f"Testing PCA components = {pca_components}")
        print("=" * 60)

        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_mnist(
            target_digit=target_digit,
            method="pca",
            pca_components=pca_components
        )

        X_val_subset = X_val[:val_subset_size]
        y_val_subset = y_val[:val_subset_size]

        for k in k_values:
            print(f"\nTesting k = {k} with PCA components = {pca_components}")

            knn = KNNLoop(k=k)
            knn.fit(X_train, y_train)

            y_val_pred = knn.predict(X_val_subset)

            acc = accuracy_score_manual(y_val_subset, y_val_pred)
            prec = precision_score_manual(y_val_subset, y_val_pred)
            rec = recall_score_manual(y_val_subset, y_val_pred)
            f1 = f1_score_manual(y_val_subset, y_val_pred)
            cm = confusion_matrix_manual(y_val_subset, y_val_pred)

            result = {
                "pca_components": pca_components,
                "k": k,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "confusion_matrix": cm
            }
            all_results.append(result)

            print(f"Accuracy : {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall   : {rec:.4f}")
            print(f"F1-score : {f1:.4f}")
            print(f"Confusion Matrix:\n{cm}")

            if best_overall is None:
                best_overall = result
            else:
                if result["f1"] > best_overall["f1"]:
                    best_overall = result
                elif result["f1"] == best_overall["f1"]:
                    if result["k"] < best_overall["k"]:
                        best_overall = result
                    elif result["k"] == best_overall["k"] and result["pca_components"] < best_overall["pca_components"]:
                        best_overall = result

    print("\n" + "=" * 60)
    print("BEST PCA + KNN SETUP")
    print("=" * 60)
    print(f"PCA components : {best_overall['pca_components']}")
    print(f"k              : {best_overall['k']}")
    print(f"Accuracy       : {best_overall['accuracy']:.4f}")
    print(f"Precision      : {best_overall['precision']:.4f}")
    print(f"Recall         : {best_overall['recall']:.4f}")
    print(f"F1-score       : {best_overall['f1']:.4f}")
    print(f"Confusion Matrix:\n{best_overall['confusion_matrix']}")
    print("=" * 60)


if __name__ == "__main__":
    main()