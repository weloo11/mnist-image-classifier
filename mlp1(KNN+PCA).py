import numpy as np

# 1. Load Data
print("I am alive!", flush=True)
# Using your paths
X_train = np.load(r"C:/Users/zeina/Downloads/ml_data/X_train_pca.npy")
X_val = np.load(r"C:/Users/zeina/Downloads/ml_data/X_val_pca.npy")
y_train = np.load(r"C:/Users/zeina/Downloads/ml_data/y_train.npy")
y_val = np.load(r"C:/Users/zeina/Downloads/ml_data/y_val.npy")

class KNNFromScratch:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        # We still loop over the validation set, but the inner loop is gone!
        for i, x in enumerate(X):
            # VECTORIZED DISTANCE: Subtraction and squaring happens for all rows at once
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # Find the indices of the k smallest distances
            k_indices = np.argsort(distances)[:self.k]
            
            # Get the labels and find the most common one
            k_nearest_labels = self.y_train[k_indices].astype(int)
            prediction = np.bincount(k_nearest_labels).argmax()
            predictions.append(prediction)
            
            # Progress bar so you know it's moving
            if (i + 1) % 100 == 0:
                print(f"Predicted {i + 1}/{len(X)} samples...", flush=True)
                
        return np.array(predictions)

# --- Metric Functions ---
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

# --- Execution ---
print("Starting KNN...", flush=True)
knn = KNNFromScratch(k=3)

# To test if it works quickly, use a slice: X_train[:2000] and X_val[:100]
knn.fit(X_train, y_train)

print("Starting predictions (Vectorized)...", flush=True)
y_val_pred = knn.predict(X_val)

print("\n--- RESULTS ---")
print("Accuracy :", accuracy_score_manual(y_val, y_val_pred))
print("Precision:", precision_score_manual(y_val, y_val_pred))
print("Recall   :", recall_score_manual(y_val, y_val_pred))
print("F1-score :", f1_score_manual(y_val, y_val_pred))
print("\nConfusion Matrix:\n", confusion_matrix_manual(y_val, y_val_pred))