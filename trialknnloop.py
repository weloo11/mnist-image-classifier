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

        # Majority vote
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


def evaluate_knn(X_train, y_train, X_val, y_val, k):
    knn = KNNLoop(k=k)
    knn.fit(X_train, y_train)
    y_val_pred = knn.predict(X_val)

    acc = accuracy_score_manual(y_val, y_val_pred)
    prec = precision_score_manual(y_val, y_val_pred)
    rec = recall_score_manual(y_val, y_val_pred)
    f1 = f1_score_manual(y_val, y_val_pred)
    cm = confusion_matrix_manual(y_val, y_val_pred)

    return {
        "k": k,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": y_val_pred
    }


def main():
    methods = ["flatten", "pca", "hog"]
    k_values = [1, 3, 5, 7, 9, 11]
    val_subset_size = 500
    target_digit = 5
    pca_components = 50

    all_results = {}
    best_overall = None

    for method in methods:
        print("\n" + "=" * 50)
        print(f"Tuning KNN with method: {method}")
        print("=" * 50)

        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_mnist(
            target_digit=target_digit,
            method=method,
            pca_components=pca_components
        )

        X_val_subset = X_val[:val_subset_size]
        y_val_subset = y_val[:val_subset_size]

        print("Training shape:", X_train.shape)
        print("Validation subset shape:", X_val_subset.shape)

        method_results = []
        best_for_method = None

        for k in k_values:
            print(f"\nTesting k = {k} for method = {method}", flush=True)

            result = evaluate_knn(
                X_train, y_train,
                X_val_subset, y_val_subset,
                k
            )

            method_results.append(result)

            print(f"Accuracy : {result['accuracy']:.4f}")
            print(f"Precision: {result['precision']:.4f}")
            print(f"Recall   : {result['recall']:.4f}")
            print(f"F1-score : {result['f1']:.4f}")
            print(f"Confusion Matrix:\n{result['confusion_matrix']}")

            if best_for_method is None:
                best_for_method = result
            else:
                if result["f1"] > best_for_method["f1"]:
                    best_for_method = result
                elif result["f1"] == best_for_method["f1"] and result["k"] < best_for_method["k"]:
                    best_for_method = result

        all_results[method] = {
            "all_k_results": method_results,
            "best": best_for_method
        }

        print("\n" + "-" * 50)
        print(f"Best result for method = {method}")
        print(f"k         : {best_for_method['k']}")
        print(f"Accuracy  : {best_for_method['accuracy']:.4f}")
        print(f"Precision : {best_for_method['precision']:.4f}")
        print(f"Recall    : {best_for_method['recall']:.4f}")
        print(f"F1-score  : {best_for_method['f1']:.4f}")
        print(f"Confusion Matrix:\n{best_for_method['confusion_matrix']}")
        print("-" * 50)

        if best_overall is None:
            best_overall = {
                "method": method,
                **best_for_method
            }
        else:
            if best_for_method["f1"] > best_overall["f1"]:
                best_overall = {
                    "method": method,
                    **best_for_method
                }
            elif best_for_method["f1"] == best_overall["f1"] and best_for_method["k"] < best_overall["k"]:
                best_overall = {
                    "method": method,
                    **best_for_method
                }

    print("\n" + "=" * 60)
    print("BEST OVERALL KNN SETUP")
    print("=" * 60)
    print(f"Method    : {best_overall['method']}")
    print(f"k         : {best_overall['k']}")
    print(f"Accuracy  : {best_overall['accuracy']:.4f}")
    print(f"Precision : {best_overall['precision']:.4f}")
    print(f"Recall    : {best_overall['recall']:.4f}")
    print(f"F1-score  : {best_overall['f1']:.4f}")
    print(f"Confusion Matrix:\n{best_overall['confusion_matrix']}")
    print("=" * 60)


if __name__ == "__main__":
    main()