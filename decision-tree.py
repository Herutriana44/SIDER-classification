import pandas as pd

df = pd.read_csv("/content/sider.csv")

feature_col = 'smiles'
label_col = [col for col in df.columns.tolist() if col not in feature_col]
label_mapping = {col: i for i, col in enumerate(label_col)}

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator # Changed import to factory function

def ecfp_fingerprint(smiles_string, radius=2, nbits=2048):
  """
  Generates Extended-Connectivity Fingerprints (ECFP) from a SMILES string.

  Args:
    smiles_string (str): The SMILES representation of the molecule.
    radius (int): The radius of the fingerprint. Default is 2.
    nbits (int): The number of bits in the fingerprint. Default is 2048.

  Returns:
    ExplicitBitVect: RDKit Extended-Connectivity Fingerprint (ECFP) as a bit vector,
                     or None if the SMILES string is invalid.
  """
  mol = Chem.MolFromSmiles(smiles_string)
  if mol is None:
    return None
  # Use MorganGenerator to avoid deprecation warning
  fgen = GetMorganGenerator(radius=radius, fpSize=nbits) # Use the factory function
  fp = fgen.GetFingerprint(mol) # Corrected method call
  return fp

df['ecfp'] = df['smiles'].apply(ecfp_fingerprint)

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, jaccard_score, hamming_loss

# Prepare features (X)
# Convert ECFP bit vectors to numpy arrays of integers (0s and 1s)
# Handle potential None values if some SMILES strings were invalid
X = np.array([list(fp) if fp is not None else np.zeros(2048) for fp in df['ecfp']])

# Prepare labels (y)
y = df[label_col].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base Decision Tree classifier
base_dtree = DecisionTreeClassifier(random_state=42)

# Wrap the base classifier in MultiOutputClassifier
multi_output_dtree = MultiOutputClassifier(estimator=base_dtree)

# Define the parameter grid for GridSearchCV
param_grid = {
    'estimator__max_depth': [None, 10, 20, 30],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__criterion': ['gini', 'entropy']
}

# Initialize GridSearchCV
# Scoring: 'jaccard' is a good choice for multi-label classification.
# 'accuracy' (exact match ratio) is also an option but often too strict.
# We use 'jaccard_weighted' if labels are imbalanced.
grid_search = GridSearchCV(
    multi_output_dtree,
    param_grid,
    cv=3, # 3-fold cross-validation
    scoring='jaccard_macro',
    n_jobs=-1, # Use all available CPU cores
    verbose=1
)

print("Starting GridSearchCV for Multi-label Decision Tree...")
grid_search.fit(X_train, y_train)
print("GridSearchCV complete.")

# Get the best estimator
best_multi_output_dtree = grid_search.best_estimator_
print(f"\nBest parameters found: {grid_search.best_params_}")

# Make predictions with the best model
y_pred = best_multi_output_dtree.predict(X_test)

# Evaluate the best model
exact_match_accuracy = accuracy_score(y_test, y_pred)
print(f"\nExact Match Accuracy (Best Model): {exact_match_accuracy:.4f}")

jaccard_similarity = jaccard_score(y_test, y_pred, average='samples')
print(f"Jaccard Similarity (average over samples, Best Model): {jaccard_similarity:.4f}")

hamming_loss_val = hamming_loss(y_test, y_pred)
print(f"Hamming Loss (Best Model): {hamming_loss_val:.4f}")

print("\nFirst 5 true labels (y_test):")
print(y_test[:5])
print("\nFirst 5 predicted labels (y_pred):")
print(y_pred[:5])

import json
import os

# Gather all experiment results into a dictionary
experiment_results = {
    'best_parameters': grid_search.best_params_,
    'exact_match_accuracy': exact_match_accuracy,
    'jaccard_similarity_samples': jaccard_similarity,
    'hamming_loss': hamming_loss_val
}

# Define the results folder and algorithm name
results_folder = 'results'
algorithm_name = 'DecisionTree' # Assuming DecisionTree was the algorithm used

# Create the results folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

# Define the output file name, including the algorithm name and folder
output_filename = os.path.join(results_folder, f"{algorithm_name}_experiment_results.json")

# Save the results to a JSON file
with open(output_filename, 'w') as f:
    json.dump(experiment_results, f, indent=4)

print(f"Experiment results successfully exported to {output_filename}")
print("Content of the exported file:")
with open(output_filename, 'r') as f:
    print(f.read())