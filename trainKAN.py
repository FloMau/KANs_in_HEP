import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import mplhep as hep
from kan import KAN
import sys
import glob
sys.path.append("./..")
from modules import plotter, KAN_plotter, KAN_trainer
from sklearn.utils import shuffle
import pickle
from hist import Hist
plt.style.use([hep.style.ROOT])
import json
import shutil
import argparse


# Set up the argument parser
parser = argparse.ArgumentParser(description='Train KAN models with specified configuration.')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')

# Parse the arguments
args = parser.parse_args()

# Load the configuration file
config_path = args.config
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Directory where the datasets are stored
input_dir = "./ML_datasets"

# Read datasets
stop = 100_000
stop_val = 33_000
df_ttH_train = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_train.csv'), nrows=stop)
df_ttH_val = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_val.csv'), nrows=int(stop_val))
df_tHq_train = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_train.csv'), nrows=stop)
df_tHq_val = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_val.csv'), nrows=int(stop_val))

train_data = pd.concat([df_ttH_train, df_tHq_train])
val_data = pd.concat([df_ttH_val, df_tHq_val])
# Labels: 0 for ttHCPEven, 1 for tHqCPEven
train_labels = np.concatenate([np.zeros(len(df_ttH_train)), np.ones(len(df_tHq_train))])
val_labels = np.concatenate([np.zeros(len(df_ttH_val)), np.ones(len(df_tHq_val))])

for col in ["phoLeadPt", "phoSubleadPt", 'met', 'leptonPt', 'topMt', "ptH", 'bJetPt', 'jetLeadPt']:
    train_data[col] = np.log(1 + train_data[col].values)
    val_data[col] = np.log(1 + val_data[col].values)

train_data, train_labels = shuffle(train_data, train_labels, random_state=0)
val_data, val_labels = shuffle(val_data, val_labels, random_state=0)

train_weights = train_data.pop("weights")
val_weights = val_data.pop("weights")

# Extract settings from the config
features_to_use = config['features_to_use']
epochs = config['epochs']
patience = config['patience']
LR = config['learning_rate']
batch_size = config['batch_size']
num_nodes = config['num_nodes']
output_dir = config['output_dir']
lamb = config["lamb"]
grid = config["grid"]
k = config["k"]
opt = config["optimizer"]
update_grid = config["update_grid"].lower() == "true"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Copy the configuration file to the output directory
shutil.copy(config_path, os.path.join(output_dir, 'config.json'))

print("Features to use:", features_to_use)
print("Epochs:", epochs)
print("Patience:", patience)
print("Learning Rate:", LR)
print("Batch Size:", batch_size)
print("Number of Nodes:", num_nodes)
print("Output Directory:", output_dir)

os.makedirs(output_dir, exist_ok=True)

# Save the list to a text file
features_file = os.path.join(output_dir, "features_to_use.txt")
with open(features_file, 'w') as f:
    for feature in features_to_use:
        f.write(f"{feature}\n")

print("INFO: training with features", features_to_use)
train_data = train_data[features_to_use]
val_data = val_data[features_to_use]

short_names = plotter.tex_feature_names
features_to_use = [short_names[f] if f in short_names.keys() else f for f in features_to_use]

# Apply standard scaling
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
val_data_scaled = scaler.transform(val_data)

# Save the scaler
scaler_filename = os.path.join(output_dir, 'scaler.pkl')
pd.to_pickle(scaler, scaler_filename)

# Convert data to PyTorch tensors
train_data_tensor = torch.tensor(train_data_scaled, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
val_data_tensor = torch.tensor(val_data_scaled, dtype=torch.float32)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1)

# Create dataset in required format for KAN
dataset = {
    'train_input': train_data_tensor,
    'train_label': train_labels_tensor,
    'test_input': val_data_tensor,
    'test_label': val_labels_tensor
}

input_dim = train_data_scaled.shape[1]
model = KAN(width=num_nodes,
            grid=grid,
            k=k,
            seed=42)

# Save the model
continueTraining = False
modelSaveName = 'model.pt' 

# Define custom BCE loss functions for training and testing
def BCE_with_sigmoid(predictions, targets):
    sigmoid_predictions = torch.sigmoid(predictions)
    loss = F.binary_cross_entropy(sigmoid_predictions, targets)
    return loss

def train_acc():
    return torch.mean(
        ((torch.sigmoid(model(dataset['train_input'])) >= 0.5).float() == dataset['train_label']).float()
    )

def test_acc():
    return torch.mean(
        ((torch.sigmoid(model(dataset['test_input'])) >= 0.5).float() == dataset['test_label']).float()
    )

model = model.speed()

results = KAN_trainer.fit_with_early_stopping_batch(
    model,
    dataset,
    opt=opt,
    lr=LR,
    steps=epochs,
    batch=batch_size,
    metrics=(train_acc, test_acc),
    loss_fn=BCE_with_sigmoid,
    lamb=lamb,
    update_grid=update_grid,
    patience=patience,
    save_path=os.path.join(output_dir, modelSaveName)
)

print(results)

# Save training history
history_df = pd.DataFrame({
    'train_loss': results['train_loss'],
    'val_loss': results['test_loss'],
    'train_accuracy': results['train_acc'],
    'val_accuracy': results['test_acc']
})
history_filename = os.path.join(output_dir, 'training_history.csv')
history_df.to_csv(history_filename, index=False)

# Plotting
epochs = range(1, len(results['train_loss']) + 1)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
font_size = 20

# Loss plot and accuracy plot
ax[0].plot(epochs, np.array(results['train_loss'])**2, label='Training Loss')
ax[0].plot(epochs, np.array(results['test_loss'])**2, label='Validation Loss')
ax[0].set_xlabel('Epochs', fontsize=font_size)
ax[0].set_ylabel('Loss', fontsize=font_size)
ax[0].set_yscale("log")
ax[0].legend(fontsize=0.8*font_size, loc="upper right")

ax[1].plot(epochs, results['train_acc'], label='Training Accuracy')
ax[1].plot(epochs, results['test_acc'], label='Validation Accuracy')
ax[1].set_xlabel('Epochs', fontsize=font_size)
ax[1].set_ylabel('Accuracy', fontsize=font_size)
ax[1].set_ylim(0.5, 1.0)
ax[1].legend(fontsize=0.8*font_size, loc="upper right")

plot_filename = os.path.join(output_dir, 'TrainingHistoryKAN.pdf') if not continueTraining else os.path.join(output_dir, 'TrainingHistoryKAN2.pdf')
fig.tight_layout()
fig.savefig(plot_filename)


### plot model
model.save_plot_data = True
model(dataset['train_input'])
plt.figure(figsize=(10,6))
model.plot(beta=100, in_vars=features_to_use, out_vars="ttH vs. tH", varscale=0.1)

if not continueTraining:
    plt.savefig(output_dir + "/KAN_model.pdf")
else:
    plt.savefig(output_dir + "/KAN_model2.pdf")
plt.close()
