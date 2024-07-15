import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist
import pickle
from scipy import integrate
from copy import deepcopy

import torch
from kan import KAN


plt.style.use([hep.style.CMS])
sys.path.append("./..")
from modules import plotter

# Dictionary with paths to separate trainings
model_dirs = {
    "MLP 128-64-32-16": "./TrainingOutputs/MLP/LowAndHighLevels/128-64-32-16/",
    "MLP 1x32": "./TrainingOutputs/MLP/LowAndHighLevels/32/",
    "MLP 1x8": "./TrainingOutputs/MLP/LowAndHighLevels/8/",
#     "KAN 25-10-5-1": "./KAN_trainingsLowLevels/AllFeatures_25_10_5_1",
    "KAN 22-1": "./TrainingOutputs/KAN/LBFGS/LowAndHighLevels/22-1-LR1e-3/",
    "KAN 22-1 g=50": "./TrainingOutputs/KAN/LBFGS/LowAndHighLevels/22-1-grid50/",
    "KAN 22-44-1": "./TrainingOutputs/KAN/LBFGS/LowAndHighLevels/22-44-1/",
}
KAN_models_dict = {
    "KAN 22-1": KAN(width=[22, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 22-44-1": KAN(width=[22, 44, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 22-1 g=50": KAN(width=[22, 1], grid=50, grid_range=(0,1), grid_eps=0, k=3, seed=42),
}

# output_dir = "./TrainingOutputs/CombinedLambdaScan/DisCoPerClass"
# output_dir = "./KANvsMLP/firstTests/"
output_dir = "./TrainingOutputs/KAN_vs_MLP/ROC_curves/"
os.makedirs(output_dir, exist_ok=True)

def human_readable_params(num):
    if num < 1000:
        return f"{num}"
    elif num < 1_000_000:
        return f"{num / 1_000:.1f}k" if num % 1_000 != 0 else f"{num // 1_000}k"
    else:
        return f"{num / 1_000_000:.1f}M" if num % 1_000_000 != 0 else f"{num // 1_000_000}M"


# Read test datasets
input_dir = "./ML_datasets"
stop = 33_000
df_ttH = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_test.csv'), nrows=stop)
df_tHq = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_test.csv'), nrows=stop)

# # Optional: Read and append validation datasets
# add_val = True
# if add_val:
#     df_ttHCPEven_val = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_val.csv'))
#     df_tHqCPEven_val = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_val.csv'))
#     df_ttHCPOdd_val = pd.read_csv(os.path.join(input_dir, 'ttHCPOdd_val.csv'))
#     df_tHqCPOdd_val = pd.read_csv(os.path.join(input_dir, 'tHqCPOdd_val.csv'))

#     df_ttH = pd.concat([df_ttH, df_ttHCPEven_val], ignore_index=True)
#     df_tHq = pd.concat([df_tHq, df_tHqCPEven_val], ignore_index=True)
#     df_ttHCPOdd_test = pd.concat([df_ttHCPOdd_test, df_ttHCPOdd_val], ignore_index=True)
#     df_tHqCPOdd_test = pd.concat([df_tHqCPOdd_test, df_tHqCPOdd_val], ignore_index=True)

ttH_weights = df_ttH.pop("weights")
tHq_weights = df_tHq.pop("weights")

test_data_dict = {
    'ttH': [df_ttH, ttH_weights],
    'tHq': [df_tHq, tHq_weights],
}

KAN_test_data_dict = deepcopy(test_data_dict)
# apply log scaling as in training
for key in KAN_test_data_dict.keys():
    for col in ['met', 'leptonPt', 'topMt', "ptH", "bJetPt", "jetLeadPt"]:
        KAN_test_data_dict[key][0][col] = np.log(1 + KAN_test_data_dict[key][0][col].values)


# Function to create histograms
def create_histogram(predictions, weights, bins=20, range=(0, 1)):
    hist = Hist.new.Reg(bins, *range, name="output").Weight()
    hist.fill(predictions, weight=weights)
    return hist

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, label, ax, n_params=None):
    if n_params is not None:
        ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f}, #{n_params})', linewidth=3)
    else:
        ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})', linewidth=3)

# Initialize figures for combined ROC curves
fig_roc_combined, ax_roc_combined = plt.subplots()

# Loop through each model directory
for model_label, model_dir in model_dirs.items():
    n_params = None
    # Read the features to use
    features_file = os.path.join(model_dir, "features_to_use.txt")
    features_to_use = []
    with open(features_file, 'r') as f:
        features_to_use = [line.strip() for line in f]

    # Load the scaler
    scaler_filename = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)

    if not "KAN" in model_dir:
        # Load the model
        model_filename = os.path.join(model_dir, 'model.h5')
        model = tf.keras.models.load_model(model_filename)

        n_params = model.count_params()
        n_params = human_readable_params(n_params)

        # Evaluate the model and create histograms
        histograms = {}
        predictions_dict = {}
        for key, [df, weights] in test_data_dict.items():
            data_scaled = scaler.transform(df[features_to_use])
            predictions = model.predict(data_scaled, batch_size=1024).flatten()
            histograms[key] = create_histogram(predictions, weights)
            predictions_dict[key] = predictions

    else:
        # Load the model
        model_filename = os.path.join(model_dir, 'model.pt')
        model = KAN_models_dict[model_label]
        model.load_state_dict(torch.load(model_filename))

        n_params = 0
        print(model.width)
        for i in range(len(model.width)-1):
            n_nodes = int(model.width[i][0])
            n_nodes_next = int(model.width[i+1][0])
            n_params += n_nodes * n_nodes_next * (model.grid + model.k)

        # Evaluate the model and create histograms
        histograms = {}
        predictions_dict = {}
        for key, [df, weights] in KAN_test_data_dict.items():
            data_scaled = scaler.transform(df[features_to_use])
            data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
            with torch.no_grad():
                output = model(data_tensor)
            predictions = torch.sigmoid(output).numpy().reshape(-1)
            histograms[key] = create_histogram(predictions, weights)
            predictions_dict[key] = predictions

    # Plotting histograms using the custom plotting function
    labels = [r"$t\bar{t}H$", r"tH"]
    process_histograms = [histograms['ttH'], histograms['tHq']]

    plotter.plot_histograms(
        hists=[hist for hist in process_histograms],
        process_labels=labels,
        output_filename=os.path.join(model_dir, f'output_distributions.pdf'),
        axis_labels=("Model output", "Density"),
        normalize=True
    )

    # Calculate and plot ROC curves

    # Calculate and plot ROC curves
    fpr, tpr, _ = roc_curve(
        y_true=np.concatenate([np.zeros(len(predictions_dict['ttH'])), np.ones(len(predictions_dict['tHq']))]),
        y_score=np.concatenate([predictions_dict['ttH'], predictions_dict['tHq']]),
        sample_weight=np.concatenate([ttH_weights, tHq_weights])
    )
    roc_auc = integrate.trapz(tpr, fpr)

    print(
        f"{model_dir = }, {roc_auc = }"
    )
    fig, ax = plt.subplots()
    plot_roc_curve(fpr, tpr, roc_auc, f"", ax, n_params)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="best")
    fig.savefig(os.path.join(model_dir, f'ROC_curve.pdf'))
    plt.close(fig)

    # Add to combined ROC plots
    plot_roc_curve(fpr, tpr, roc_auc, model_label, ax_roc_combined, n_params)

# Finalize and save combined ROC curves
ax_roc_combined.set_xlabel("False Positive Rate")
ax_roc_combined.set_ylabel("True Positive Rate")
ax_roc_combined.set_ylim(0.4, 1.01)
ax_roc_combined.legend(loc="best")
fig_roc_combined.tight_layout()
fig_roc_combined.savefig(os.path.join(output_dir, 'combined_ROC_curves.pdf'))

plt.close(fig_roc_combined)
