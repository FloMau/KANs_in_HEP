import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist
import pickle
from scipy import integrate
from copy import deepcopy
from sklearn.utils import resample
import torch
from kan import KAN


plt.style.use([hep.style.CMS])
sys.path.append("./..")
from modules import plotter

model_dirs = {
    # "KAN 22-1": "../TrainingOutputs/KANs26Jul2024/22-1-LBFGS/",
    "KAN 22-1": "../TrainingOutputs/KANs26Jul2024/22-1-Adam/",
    # "KAN 22-44-1": "../TrainingOutputs/KANs26Jul2024/22-44-1-LBFGS/",
    "KAN 22-44-1": "../TrainingOutputs/KANs26Jul2024/22-44-1-Adam/",
    # "KAN 22-10-5-2-1": "../TrainingOutputs/KANs26Jul2024/22-10-5-2-1-LBFGS/",
    "KAN 22-10-5-2-1 Adam": "../TrainingOutputs/KANs26Jul2024/22-10-5-2-1-Adam/",
    # "KAN 22-1 g=50": "../TrainingOutputs/KANs26Jul2024/22-1-LBFGS-grid50/",
}


KAN_models_dict = {
    "KAN 22-1": KAN(width=[22, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 22-1 Adam": KAN(width=[22, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 22-44-1": KAN(width=[22, 44, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 22-44-1 Adam": KAN(width=[22, 44, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 22-1 g=50": KAN(width=[22, 1], grid=50, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 22-10-5-2-1": KAN(width=[22, 10, 5, 2, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 22-10-5-2-1 Adam": KAN(width=[22, 10, 5, 2, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
}


# Read test datasets
input_dir = "../ML_datasets"
stop = 100_000
df_ttH = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_test.csv'), nrows=stop)
df_tHq = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_test.csv'), nrows=stop)

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


# Loop through each model directory
histograms_dict = {}
for model_label, model_dir in model_dirs.items():

    features_file = os.path.join(model_dir, "features_to_use.txt")
    with open(features_file, 'r') as f:
        features_to_use = [line.strip() for line in f]

    scaler_filename = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)

    model_filename = os.path.join(model_dir, 'model.pt')
    model = KAN_models_dict[model_label]
    model.load_state_dict(torch.load(model_filename))

    histograms_dict[model_label] = {}
    for key, (df, weights) in KAN_test_data_dict.items():
        data_scaled = scaler.transform(df[features_to_use])
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
        with torch.no_grad():
            output = model(data_tensor)
        predictions = torch.sigmoid(output).numpy().reshape(-1)
        histograms_dict[model_label][key] = create_histogram(predictions, weights)


# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey='row')
handles, _labels = [], []
for idx, model_label in enumerate(model_dirs.keys()):
    
    ax = axes[idx]
    label = "Model output"
    hist_ttH = histograms_dict[model_label]["ttH"]
    hist_tHq = histograms_dict[model_label]["tHq"]
    _, ax = plotter.plot_histograms(
        hists=[hist_ttH, hist_tHq],
        process_labels=[r"$\mathit{t\bar{t}H}$", r"$\mathit{tH}$"],
        ax=ax,
        normalize=True,
        axis_labels=(label, 'Fraction of events' if idx == 0 else ''),
        include_flow=True,
        return_figure=True
    )
    # Collect the handles and labels for the shared legend
    handles, _labels = ax.get_legend_handles_labels()
    # handles.extend(h)
    # _labels.extend(l)
    # Remove the legend from the current subplot
    binning = hist_ttH.to_numpy()[1]
    ax.set_xlim(binning[0], binning[-1])
    ax.get_legend().remove()
    # ax.set_xlim(-0.02, 1.02)
    for line in ax.get_lines():
        line.set_linewidth(4)
    for patch in ax.patches:
        patch.set_linewidth(4)
    for collection in ax.collections:
        collection.set_linewidth(4)

ax.set_ylim(0.001, ax.get_ylim()[1])

fontsize = 30
for ax in axes.flatten():
    ax.xaxis.label.set_size(fontsize)  # Set the x-axis label fontsize
    ax.yaxis.label.set_size(fontsize)  # Set the y-axis label fontsize


# axes.flatten()[0].set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])  # Replace (0, 1) with your desired ylim
# axes.flatten()[0].set_ylim(axes.flatten()[0].get_ylim()[0], 0.9*axes.flatten()[0].get_ylim()[1]) 
# axes.flatten()[5].set_ylim(axes.flatten()[5].get_ylim()[0], 0.71*axes.flatten()[5].get_ylim()[1]) 

titles = ['KAN 22-1', 'KAN 22-44-1', 'KAN 22-10-5-2-1']
for i, ax in enumerate(axes):
    ax.set_title(titles[i], fontsize=fontsize+2)
# Create a single shared legend
fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to make space for the legend
fig.legend(handles, _labels, loc='upper right', ncol=2, fontsize=fontsize+2, bbox_to_anchor=(0.97, 1.05))
# Adjust layout and save

# Adjust layout and save
os.makedirs("./PlotsOutputDistribution", exist_ok=True)
fig.savefig("./PlotsOutputDistribution/outputDistributions.pdf")
plt.close(fig)
