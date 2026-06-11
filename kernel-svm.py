import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, jaccard_score, hamming_loss
import json
import os

# Load data
df = pd.read_csv("sider.csv")

feature_col = 'smiles'
label_col = [col for col in df.columns.tolist() if col not in feature_col]

def ecfp_fingerprint(smiles_string, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    fgen = GetMorganGenerator(radius=radius, fpSize=nbits)
    fp = fgen.GetFingerprint(mol)
    return fp

df['ecfp'] = df['smiles'].apply(ecfp_fingerprint)

# Prepare features (X)
X = np.array([list(fp) if fp is not None else np.zeros(2048) for fp in df['ecfp']])

# Prepare labels (y)
y = df[label_col].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base SVM classifier with RBF kernel
base_svm = SVC(kernel='rbf', probability=True, random_state=42)

# Wrap the base classifier in MultiOutputClassifier
multi_output_svm = MultiOutputClassifier(estimator=base_svm)

# Define the parameter grid
param_grid = {
    'estimator__C': [0.1, 1.0, 10.0],
    'estimator__gamma': ['scale', 'auto', 0.1]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    multi_output_svm,
    param_grid,
    cv=3,
    scoring='jaccard_macro',
    n_jobs=-1,
    verbose=1
)

print("Starting GridSearchCV for Multi-label Kernel SVM...")
grid_search.fit(X_train, y_train)
print("GridSearchCV complete.")

# Best estimator and evaluation
best_multi_output_svm = grid_search.best_estimator_
y_pred = best_multi_output_svm.predict(X_test)

exact_match_accuracy = accuracy_score(y_test, y_pred)
jaccard_similarity = jaccard_score(y_test, y_pred, average='samples')
hamming_loss_val = hamming_loss(y_test, y_pred)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
print(f"Jaccard Similarity: {jaccard_similarity:.4f}")
print(f"Hamming Loss: {hamming_loss_val:.4f}")

# Export results
experiment_results = {
    'best_parameters': grid_search.best_params_,
    'exact_match_accuracy': exact_match_accuracy,
    'jaccard_similarity_samples': jaccard_similarity,
    'hamming_loss': hamming_loss_val
}

results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)
output_filename = os.path.join(results_folder, "KernelSVM_experiment_results.json")

with open(output_filename, 'w') as f:
    json.dump(experiment_results, f, indent=4)

print(f"Experiment results exported to {output_filename}")
