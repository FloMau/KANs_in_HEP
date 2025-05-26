import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import log_loss
import torch
from kan import KAN
import json
from functools import partial

# Ensure that any custom modules can be imported
sys.path.append("./..")

from modules import plotter, KAN_plotter  # Import KAN_plotter for get_L1_norms function

# Define the path to your second deepest MLP model
# model_dir = "../TrainingOutputs/MLPs26Jul2024/128-64-32-16/"
model_dir = "../TrainingOutputs/MLPs26Jul2024/256-128-64-32-16/"
# model_dir = "../TrainingOutputs/MLPs26Jul2024/8/"

# Paths to the trained KAN models
kan_model_dir1 = "../TrainingOutputs/KANs26Jul2024/22-1-Adam/"
kan_model_dir2 = "../TrainingOutputs/KANs26Jul2024/22-45-1-Adam/"

# Load the list of features used during MLP training
features_file = os.path.join(model_dir, "features_to_use.txt")
with open(features_file, 'r') as f:
    features_to_use_mlp = [line.strip() for line in f]

# Load the scaler used during MLP training
scaler_filename = os.path.join(model_dir, 'scaler.pkl')
with open(scaler_filename, 'rb') as f:
    scaler_mlp = pickle.load(f)

# Load your trained MLP model
model_filename = os.path.join(model_dir, 'model.h5')
model_mlp = tf.keras.models.load_model(model_filename)

# Read the test dataset
input_dir = "../ML_datasets"
stop = 2000  # Number of rows to read
df_ttH = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_test.csv'), nrows=stop)
df_tHq = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_test.csv'), nrows=stop)

# Assign labels to the datasets
df_ttH['label'] = 0  # ttH is class 0
df_tHq['label'] = 1  # tHq is class 1

# Combine the datasets
df_test = pd.concat([df_ttH, df_tHq], ignore_index=True)

# Remove weights column if present
if 'weights' in df_test.columns:
    df_test = df_test.drop(columns=['weights'])

# Select features and labels
X_test_mlp = df_test[features_to_use_mlp]
y_test = df_test['label'].values

# Scale the data using the loaded MLP scaler
X_test_scaled_mlp = scaler_mlp.transform(X_test_mlp)

# Optional: Select a subset to reduce computation time
X_sample_mlp = X_test_scaled_mlp[:10000]  # You can adjust the number of samples here

# Compute SHAP values for MLP
# Initialize the GradientExplainer
explainer = shap.DeepExplainer(model_mlp, X_sample_mlp)

# Compute SHAP values for the test data
shap_values = explainer.shap_values(X_sample_mlp)

# Since it's a binary classification model, shap_values is an array of shape (num_samples, num_features, num_outputs)
# We need to index appropriately to get the SHAP values for the output neuron
shap_values = shap_values[:, :, 0]

# Compute mean absolute SHAP values for each feature
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# Compute baseline loss for MLP
predictions = model_mlp.predict(X_test_scaled_mlp).flatten()
baseline_loss = log_loss(y_test, predictions)

# Compute permutation importance for MLP
permutation_importances_mlp = []
for i in range(len(features_to_use_mlp)):
    X_permuted = X_test_scaled_mlp.copy()
    rng = np.random.RandomState(42 + i)  # For reproducibility
    X_permuted[:, i] = rng.permutation(X_permuted[:, i])
    permuted_predictions = model_mlp.predict(X_permuted).flatten()
    permuted_loss = log_loss(y_test, permuted_predictions)
    importance = permuted_loss - baseline_loss
    permutation_importances_mlp.append(importance)

# Convert to numpy array
permutation_importances_mlp = np.array(permutation_importances_mlp)

# Normalize the SHAP values and permutation importances
mean_abs_shap = mean_abs_shap / np.max(mean_abs_shap)
permutation_importances_mlp = permutation_importances_mlp / np.max(permutation_importances_mlp)



# Now, process the KAN models


# Load the configuration file for KAN model 1
kan_config_path1 = os.path.join(kan_model_dir1, "config.json")
with open(kan_config_path1, 'r') as config_file:
    kan_config1 = json.load(config_file)
kan_features_to_use = kan_config1['features_to_use']

# Load the configuration file for KAN model 2
kan_config_path2 = os.path.join(kan_model_dir2, "config.json")
with open(kan_config_path2, 'r') as config_file:
    kan_config2 = json.load(config_file)

# Read the test dataset for KAN models
# Apply the same preprocessing as during KAN training
df_ttH_test_kan = df_ttH.copy()
df_tHq_test_kan = df_tHq.copy()

# Apply log transformation to certain columns as in KAN training
for col in ["phoLeadPt", "phoSubleadPt", 'met', 'leptonPt', 'topMt', "ptH", 'bJetPt', 'jetLeadPt']:
    df_ttH_test_kan[col] = np.log(1 + df_ttH_test_kan[col].values)
    df_tHq_test_kan[col] = np.log(1 + df_tHq_test_kan[col].values)

# Combine the datasets
df_test_kan = pd.concat([df_ttH_test_kan, df_tHq_test_kan], ignore_index=True)
df_test_kan = df_test_kan.sample(frac=1, random_state=42).reset_index(drop=True) # because for the KAN we cannot evaluate SHAP on the whole dataset...
y_test_kan = df_test_kan['label'].values

# Remove weights column if present
if 'weights' in df_test_kan.columns:
    df_test_kan = df_test_kan.drop(columns=['weights'])

# Select features for KAN models
X_test_kan = df_test_kan[kan_features_to_use]

scaler_filename = os.path.join(kan_model_dir1, 'scaler.pkl')
with open(scaler_filename, 'rb') as f:
    scaler_kan = pickle.load(f)
X_test_scaled_kan = scaler_kan.transform(X_test_kan)

# Convert to PyTorch tensors
X_test_kan_tensor = torch.tensor(X_test_scaled_kan, dtype=torch.float32)
y_test_kan_tensor = torch.tensor(y_test_kan, dtype=torch.float32).unsqueeze(1)

# Build and load KAN model 1
model_kan1 = KAN(
    width=kan_config1['num_nodes'],
    grid=kan_config1['grid'],
    k=kan_config1['k'],
    seed=42
)
model_kan1.load_state_dict(torch.load(os.path.join(kan_model_dir1, "model.pt")))

# Apply the model to the data to set the spline values
with torch.no_grad():
    outputs_kan1 = model_kan1(X_test_kan_tensor)
    predictions_kan1 = torch.sigmoid(outputs_kan1).cpu().numpy().flatten()



###### tricks

def KAN_numpy_wrapper(model, input_numpy):
    # Convert NumPy array to torch.Tensor
    input_tensor = torch.from_numpy(input_numpy).float()
    # Ensure the tensor is on the same device as your model
    input_tensor = input_tensor.to(next(model.parameters()).device)
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Get model output
        output_tensor = model(input_tensor)
        # Apply sigmoid activation if necessary
        output_tensor = torch.sigmoid(output_tensor)
    # Convert output tensor to NumPy array
    output_numpy = output_tensor.cpu().numpy()
    return output_numpy

n_shap_kan = 200

# Convert torch.Tensor to NumPy array
X_test_kan_numpy = X_test_kan_tensor.cpu().numpy()
# Initialize the explainer with the wrapped model and background data
model_kan1_wrapper = partial(KAN_numpy_wrapper, model_kan1)
explainer = shap.SamplingExplainer(model_kan1_wrapper, X_test_kan_numpy[:50])
# Compute SHAP values for a subset of your data
shap_values_kan1 = explainer.shap_values(X_test_kan_numpy[:n_shap_kan])

# # # Initialize the GradientExplainer
# explainer = shap.SamplingExplainer(model_kan1, X_test_kan_numpy[:100])
# # Compute SHAP values for the test data
# shap_values_kan1 = explainer.shap_values(X_test_kan_numpy[:n_shap_kan])

# shap_values_kan1 = shap_values_kan1[:, 0]
# Compute mean absolute SHAP values for each feature
mean_abs_shap_kan1 = np.mean(np.abs(shap_values_kan1), axis=0)

mean_abs_shap_kan1_normalized = mean_abs_shap_kan1 / np.max(mean_abs_shap_kan1)

# Compute baseline loss for KAN model 1
baseline_loss_kan1 = log_loss(y_test_kan, predictions_kan1)

# Compute permutation importance for KAN model 1
permutation_importances_kan1 = []
for i in range(len(kan_features_to_use)):
    X_permuted = X_test_kan_tensor.clone()
    rng = np.random.RandomState(42 + i)  # For reproducibility
    permuted_indices = rng.permutation(len(X_test_kan_tensor))
    X_permuted[:, i] = X_permuted[permuted_indices, i]
    with torch.no_grad():
        outputs_permuted = model_kan1(X_permuted)
        predictions_permuted = torch.sigmoid(outputs_permuted).cpu().numpy().flatten()
    permuted_loss = log_loss(y_test_kan, predictions_permuted)
    importance = permuted_loss - baseline_loss_kan1
    permutation_importances_kan1.append(importance)

permutation_importances_kan1 = np.array(permutation_importances_kan1)
permutation_importances_kan1_normalized = permutation_importances_kan1 / np.max(permutation_importances_kan1)

# Build and load KAN model 2
model_kan2 = KAN(
    width=kan_config2['num_nodes'],
    grid=kan_config2['grid'],
    k=kan_config2['k'],
    seed=42
)
model_kan2.load_state_dict(torch.load(os.path.join(kan_model_dir2, "model.pt")))

# Apply the model to the data to set the spline values
with torch.no_grad():
    outputs_kan2 = model_kan2(X_test_kan_tensor)
    predictions_kan2 = torch.sigmoid(outputs_kan2).cpu().numpy().flatten()



# Initialize the explainer with the wrapped model and background data
model_kan2_wrapper = partial(KAN_numpy_wrapper, model_kan2)
explainer = shap.SamplingExplainer(model_kan2_wrapper, X_test_kan_numpy[:50])
# Compute SHAP values for a subset of your data
shap_values_kan2 = explainer.shap_values(X_test_kan_numpy[:n_shap_kan])

# Initialize the GradientExplainer
# explainer = shap.SamplingExplainer(model_kan2, X_test_kan_tensor[:500])
# # Compute SHAP values for the test data
# shap_values_kan2 = explainer.shap_values(X_test_kan_tensor[:500])
# shap_values_kan2 = shap_values_kan2[:, :, 0]
# Compute mean absolute SHAP values for each feature

mean_abs_shap_kan2 = np.mean(np.abs(shap_values_kan2), axis=0)
mean_abs_shap_kan2_normalized = mean_abs_shap_kan2 / np.max(mean_abs_shap_kan2)

# Compute baseline loss for KAN model 2
baseline_loss_kan2 = log_loss(y_test_kan, predictions_kan2)

# Compute permutation importance for KAN model 2
permutation_importances_kan2 = []
for i in range(len(kan_features_to_use)):
    X_permuted = X_test_kan_tensor.clone()
    rng = np.random.RandomState(42 + i)  # For reproducibility
    permuted_indices = rng.permutation(len(X_test_kan_tensor))
    X_permuted[:, i] = X_permuted[permuted_indices, i]
    with torch.no_grad():
        outputs_permuted = model_kan2(X_permuted)
        predictions_permuted = torch.sigmoid(outputs_permuted).cpu().numpy().flatten()
    permuted_loss = log_loss(y_test_kan, predictions_permuted)
    importance = permuted_loss - baseline_loss_kan2
    permutation_importances_kan2.append(importance)

permutation_importances_kan2 = np.array(permutation_importances_kan2)
permutation_importances_kan2_normalized = permutation_importances_kan2 / np.max(permutation_importances_kan2)

# Use get_L1_norms to obtain L1 norms from KAN models
L1_norms_kan1 = KAN_plotter.get_L1_norms(model_kan1)
L1_norms_kan2 = KAN_plotter.get_L1_norms(model_kan2)

# Extract norms of the first layer
First_layer_norms_kan1 = L1_norms_kan1[0]  # Shape: (n_features, n_next)
First_layer_norms_kan2 = L1_norms_kan2[0]

# Sum over the outputs to get total importance per feature
Total_norms_kan1 = np.sum(First_layer_norms_kan1, axis=1)
Total_norms_kan2 = np.sum(First_layer_norms_kan2, axis=1)

# Normalize the L1 norms
Total_norms_kan1_normalized = Total_norms_kan1 / np.max(Total_norms_kan1)
Total_norms_kan2_normalized = Total_norms_kan2 / np.max(Total_norms_kan2)

# Reorder the MLP feature importance values to match the order of KAN features
# Create a mapping from MLP features to indices
feature_to_index_mlp = {feature: idx for idx, feature in enumerate(features_to_use_mlp)}

# Get indices of KAN features in the MLP feature order
indices_in_mlp_order = [feature_to_index_mlp[feature] for feature in kan_features_to_use]

# Reorder MLP feature importance arrays
mean_abs_shap_reordered = mean_abs_shap[indices_in_mlp_order]
permutation_importances_mlp_reordered = permutation_importances_mlp[indices_in_mlp_order]

# Use tex feature names as in the KAN for the x-axis ticks
short_names = plotter.tex_feature_names
feature_labels = [short_names[f] if f in short_names.keys() else f for f in kan_features_to_use]

# Create a bar plot comparing SHAP values, permutation importance, and L1 norms from KAN models
n_features = len(kan_features_to_use)
indices = np.arange(n_features)


width = 0.12  # Adjusted width to accommodate more bars
plt.figure(figsize=(20, 8))

# Plot mean absolute SHAP values
plt.bar(indices - 2*width, mean_abs_shap_reordered, width=width, label='Mean Absolute SHAP Values (MLP)')

# Plot permutation importances for MLP
plt.bar(indices - width, permutation_importances_mlp_reordered, width=width, label='Permutation Importance (MLP)')

# Plot permutation importances for KAN model 1
plt.bar(indices, permutation_importances_kan1_normalized, width=width, label='Permutation Importance (KAN 22-1)')

# Plot permutation importances for KAN model 2
plt.bar(indices + width, permutation_importances_kan2_normalized, width=width, label='Permutation Importance (KAN 22-45-1)')

# Plot L1 norms from KAN model 1
plt.bar(indices + 2*width, Total_norms_kan1_normalized, width=width, label='L1 Norms (KAN 22-1)')

# Plot L1 norms from KAN model 2
plt.bar(indices + 3*width, Total_norms_kan2_normalized, width=width, label='L1 Norms (KAN 22-45-1)')

plt.xticks(indices, feature_labels, rotation='vertical')
plt.ylabel('Feature Importance (Normalized)')
plt.legend(loc='upper right')
plt.tight_layout()

# Save the plot
output_dir = "./featureImportanceComparison/"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'feature_importance_comparison.pdf'))
plt.close()




plt.figure(figsize=(20, 8))
width = 0.2
# Plot mean absolute SHAP values
plt.bar(indices - width, mean_abs_shap_reordered, width=width, label='Deep MLP')

# Plot L1 norms from KAN model 1
plt.bar(indices, mean_abs_shap_kan1_normalized, width=width, label='KAN 22-1')

# Plot L1 norms from KAN model 2
plt.bar(indices + width, mean_abs_shap_kan2_normalized, width=width, label='KAN 22-45-1')

plt.xticks(indices, feature_labels, rotation='vertical')
plt.ylabel('Mean Absolute SHAP Values')
# plt.title('Feature Importance Comparison')
plt.legend(loc='upper right')
plt.tight_layout()

# Save the plot
output_dir = "./featureImportanceComparison/"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'SHAP_comparison.pdf'))
plt.close()



plt.figure(figsize=(20, 8))
width = 0.15
# Plot permutation importances for KAN model 1
plt.bar(indices - 1.5*width, permutation_importances_kan1_normalized, width=width, label='Permutation Importance (KAN 22-1)')

# Plot permutation importances for KAN model 2
plt.bar(indices - 0.5*width, permutation_importances_kan2_normalized, width=width, label='Permutation Importance (KAN 22-45-1)')

# Plot L1 norms from KAN model 1
plt.bar(indices + 0.5*width, Total_norms_kan1_normalized, width=width, label='L1 Norms (KAN 22-1)')

# Plot L1 norms from KAN model 2
plt.bar(indices + 1.5*width, Total_norms_kan2_normalized, width=width, label='L1 Norms (KAN 22-45-1)')

plt.xticks(indices, feature_labels, rotation='vertical')
plt.ylabel('Feature Importance (normalized)')
plt.legend(loc='upper right')
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_dir, 'L1_vs_permutation.pdf'))
plt.close()


plt.figure(figsize=(20, 8))
width = 0.15
# Plot permutation importances for KAN model 1
plt.bar(indices - 1.5*width, mean_abs_shap_kan1_normalized, width=width, label='Mean abs. SHAP (KAN 22-1)')

# Plot permutation importances for KAN model 2
plt.bar(indices - 0.5*width, mean_abs_shap_kan2_normalized, width=width, label='Mean abs. SHAP (KAN 22-45-1)')

# Plot L1 norms from KAN model 1
plt.bar(indices + 0.5*width, Total_norms_kan1_normalized, width=width, label='L1 norms (KAN 22-1)')

# Plot L1 norms from KAN model 2
plt.bar(indices + 1.5*width, Total_norms_kan2_normalized, width=width, label='L1 norms (KAN 22-45-1)')

plt.xticks(indices, feature_labels, rotation='vertical')
plt.ylabel('Feature Importance (normalized)')
plt.legend(loc='upper right')
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_dir, 'L1_vs_SHAP.pdf'))
plt.close()




plt.figure(figsize=(20, 8))
width = 0.2
# Plot permutation importances for MLP
plt.bar(indices - width, permutation_importances_mlp_reordered, width=width, label='Permutation Importance (Deep MLP)')

# Plot permutation importances for KAN model 1
plt.bar(indices, permutation_importances_kan1_normalized, width=width, label='Permutation Importance (KAN 22-1)')

# Plot permutation importances for KAN model 2
plt.bar(indices + width, permutation_importances_kan2_normalized, width=width, label='Permutation Importance (KAN 22-45-1)')


plt.xticks(indices, feature_labels, rotation='vertical')
plt.ylabel('Feature importance (normalized)')
plt.legend(loc='upper right')
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_dir, 'permutationComparison.pdf'))
plt.close()


plt.figure(figsize=(20, 8))
width = 0.2
# Plot permutation importances for MLP
plt.bar(indices - width, Total_norms_kan1_normalized, width=width, label='L1 Norms (KAN 22-1)')

# Plot permutation importances for KAN model 1
plt.bar(indices, permutation_importances_kan1_normalized, width=width, label='Permutation Importance (KAN 22-1)')

# Plot permutation importances for KAN model 2
plt.bar(indices + width, mean_abs_shap_kan1_normalized, width=width, label='Mean abs. SHAP (KAN 22-1)')


plt.xticks(indices, feature_labels, rotation='vertical')
plt.ylabel('Feature importance (normalized)')
plt.legend(loc='upper right')
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_dir, 'KAN_22-1.pdf'))
plt.close()



plt.figure(figsize=(20, 8))
width = 0.2
# Plot permutation importances for MLP
plt.bar(indices - width, Total_norms_kan2_normalized, width=width, label='L1 Norms (KAN 22-45-1)')

# Plot permutation importances for KAN model 1
plt.bar(indices, permutation_importances_kan2_normalized, width=width, label='Permutation Importance (KAN 22-45-1)')

# Plot permutation importances for KAN model 2
plt.bar(indices + width, mean_abs_shap_kan2_normalized, width=width, label='Mean abs. SHAP (KAN 22-45-1)')


plt.xticks(indices, feature_labels, rotation='vertical')
plt.ylabel('Feature importance (normalized)')
plt.legend(loc='upper right')
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_dir, 'KAN_22-45-1.pdf'))
plt.close()
