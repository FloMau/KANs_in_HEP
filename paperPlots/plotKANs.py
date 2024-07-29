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

model_dir = "../TrainingOutputs/KANs26Jul2024/22-1-Adam/"
model2_dir = "../TrainingOutputs/KANs26Jul2024/22-3-1-Adam/"
output_dir = "./KAN_plots/"

# Load the configuration file
config_path = model_dir + "config.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Directory where the datasets are stored
input_dir = "../ML_datasets"

# Read datasets
stop = 10_000
stop = None
df_ttH_train = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_train.csv'), nrows=stop)
df_tHq_train = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_train.csv'), nrows=stop)

train_data = pd.concat([df_ttH_train, df_tHq_train])
# Labels: 0 for ttHCPEven, 1 for tHqCPEven
train_labels = np.concatenate([np.zeros(len(df_ttH_train)), np.ones(len(df_tHq_train))])

for col in ["phoLeadPt", "phoSubleadPt", 'met', 'leptonPt', 'topMt', "ptH", 'bJetPt', 'jetLeadPt']:
    train_data[col] = np.log(1 + train_data[col].values)

train_weights = train_data.pop("weights")

# Extract settings from the config
features_to_use = config['features_to_use']
epochs = config['epochs']
patience = config['patience']
LR = config['learning_rate']
batch_size = config['batch_size']
num_nodes = config['num_nodes']
lamb = config["lamb"]
grid = config["grid"]
k = config["k"]
opt = config["optimizer"]
update_grid = config["update_grid"].lower() == "true"


print("Features to use:", features_to_use)
print("Epochs:", epochs)
print("Patience:", patience)
print("Learning Rate:", LR)
print("Batch Size:", batch_size)
print("Number of Nodes:", num_nodes)
print("Output Directory:", output_dir)

os.makedirs(output_dir, exist_ok=True)

print("INFO: training with features", features_to_use)
train_data = train_data[features_to_use]

short_names = plotter.tex_feature_names
feature_labels = [short_names[f] if f in short_names.keys() else f for f in features_to_use]

# Apply standard scaling
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)


# Convert data to PyTorch tensors
train_data_tensor = torch.tensor(train_data_scaled, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)

# Create dataset in required format for KAN
dataset = {
    'train_input': train_data_tensor,
    'train_label': train_labels_tensor,
}

input_dim = train_data_scaled.shape[1]
model = KAN(width=num_nodes,
            grid=grid,
            # grid_range=(0,1),
            # grid_eps=0.,
            k=k,
            seed=42)


model.load_state_dict(torch.load(model_dir + "model.pt"))
model(dataset["train_input"])

# print(model.edge_scores)
# exit()
plot_model = False
if plot_model:
    plt.figure(figsize=(10,6))
    KAN_plotter.plot(model, folder=output_dir+"Splines", beta=3, mask=False, metric='backward', scale=1, tick=True, sample=False, in_vars=feature_labels, out_vars=['Output'], title=None, varscale=0.22)
    # # KAN_plotter.plot_custom(model, beta=50, in_vars=features_to_use, folder=output_dir+"Splines/", out_vars=['ttH vs. tH'], title = 'My KAN', scale=10, tick=True, tick_size=100)
    # plt.tight_layout()
    plt.savefig(output_dir + "/KAN_22-1.pdf", bbox_inches='tight')
    plt.close()



model2 = KAN(width=[22,3,1],
        grid=grid,
        # grid_range=(0,1),
        # grid_eps=0.,
        k=k,
        seed=42)

model2.load_state_dict(torch.load(model2_dir + "model.pt"))
model2(dataset["train_input"])
plot_model2 = False
if plot_model2:

    # print(model2.edge_scores)
    # exit()

    plt.figure(figsize=(10,6))
    KAN_plotter.plot(model2, folder=output_dir+"Splines", beta=3, mask=False, metric='backward', scale=1, tick=True, sample=False, in_vars=feature_labels, out_vars=['Output'], title=None, varscale=0.22)
    # # KAN_plotter.plot_custom(model, beta=50, in_vars=features_to_use, folder=output_dir+"Splines/", out_vars=['ttH vs. tH'], title = 'My KAN', scale=10, tick=True, tick_size=100)
    # plt.tight_layout()
    plt.savefig(output_dir + "/KAN_22-3-1.pdf", bbox_inches='tight')
    plt.close()


dataset_ttH = train_data_scaled[train_labels == 0]
dataset_tHq = train_data_scaled[train_labels == 1]
weights_ttH = train_weights[train_labels == 0]
weights_tHq = train_weights[train_labels == 1]

plot_scaled_hists = True
if plot_scaled_hists:

    features_to_include = ["nLeptons", "nJets", "DeltaY_Hb", "delta_eta_b_j", "DeltaY_Hl"]
    if not plot_model:
        model.attribute()
    if not plot_model2:
        model2.attribute()

    # Plotting
    fig, axes = plt.subplots(5, 2, figsize=(13, 20)) #, sharey='row')
    handles, _labels = [], []
    idx = 0
    for i, fea in enumerate(features_to_use):

        if fea not in features_to_include:
            continue

        ax = axes[idx, 0]
        right_ax = axes[idx, 1]
        idx += 1

        n_bins = 20
        if "njets" in fea.lower() or "nlepton" in fea.lower():
            # print(np.unique(train_data_scaled[:,i]))
            n_bins = len(np.unique(train_data_scaled[:,i]))
            if "jets" in fea.lower():
                n_bins += 1
        hist_ttH = Hist.new.Reg(n_bins, 0, 1).Weight()
        hist_tHq = Hist.new.Reg(n_bins, 0, 1).Weight()
        hist_ttH.fill(
            dataset_ttH[:,i], weight=weights_ttH
        )
        hist_tHq.fill(
            dataset_tHq[:,i], weight=weights_tHq
        )

        label = plotter.tex_feature_names[fea]
        hep.histplot(
            [hist_ttH, hist_tHq],
            label=[r"$\mathit{t\bar{t}H}$", r"$\mathit{tH}$"],
            # bins=binning,
            linewidth=4,
            yerr=True,
            ax=ax,
            # linestyle=
            density=True
        )
        binning = hist_ttH.to_numpy()[1]
        ax.set_xlim(binning[0], binning[-1])

        for line in ax.get_lines():
            line.set_linewidth(4)
        for patch in ax.patches:
            patch.set_linewidth(4)
        for collection in ax.collections:
            collection.set_linewidth(4)

        ### add likelihood ratios:
        values_ttH, edges_ttH = hist_ttH.values(), hist_ttH.axes[0].edges
        values_tHq, edges_tHq = hist_tHq.values(), hist_tHq.axes[0].edges

        # Calculate the bin widths
        bin_width = np.diff(edges_ttH)

        # Normalize the histogram values to get the density
        density_ttH = values_ttH / (np.sum(weights_ttH) * bin_width)
        density_tHq = values_tHq / (np.sum(weights_tHq) * bin_width)

        bins = hist_ttH.to_numpy()[1]
        bin_centers = [(bins[ib] + bins[ib+1])/2 for ib in range(len(bins)-1)]

        ratio = np.log(density_tHq / density_ttH)
        ratio = np.nan_to_num(ratio, nan=5, neginf=-5, posinf=5)

        ax2 = ax.twinx()
        ax2.plot(
            bin_centers, 
            ratio, 
            color="tab:grey",
            lw=5,
            label=r"Log. likelihood ratio $\mathit{tH / t\bar{t}H}$",
            linestyle="dashed",
            marker="none",
        )

        handles_left, labels_left = ax.get_legend_handles_labels()
        handles_left2, labels_left2 = ax2.get_legend_handles_labels()

        ### plots of trained splines:
        l = 0
        width_out2 = np.array(model2.width_out)
        n_next2 = width_out2[l + 1]
        linestyles = ["solid", "dashed", "dotted"]
        for j in range(n_next2):
            rank = torch.argsort(model2.acts[l][:, i])

            right_ax.plot(model2.acts[l][:, i][rank].cpu().detach().numpy(), 
                model2.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), 
                color="tab:green", lw=4, label=f"22-3-1 spline {j+1}", linestyle=linestyles[j]
            )
        width_out = np.array(model.width_out)
        n_next = width_out[l + 1]

        for j in range(n_next):
            rank = torch.argsort(model.acts[l][:, i])

            right_ax.plot(model.acts[l][:, i][rank].cpu().detach().numpy(), 
                model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), 
                color="tab:red", lw=4, label="22-1 spline", linestyle=linestyles[j]
            )
        handles_right, labels_right = right_ax.get_legend_handles_labels()
        # ax.legend(lines + lines2, labels + labels2)
        fontsize = 28
        ax.set_xlabel("Scaled " + label, fontsize=fontsize)
        ax.set_ylabel("Density", fontsize=fontsize)
        right_ax.set_xlabel("Scaled " + label, fontsize=fontsize)
        right_ax.set_ylabel("Spline", fontsize=fontsize)
        # ax2.set_ylabel(r"$ln\left(\mathit{p(tH) / p(t\bar{t}H}\right)$", fontsize=fontsize)
        ax2.set_ylabel(r"Log. LH ratio", fontsize=fontsize)

    # Create invisible axes for the legends
    legend_ax_left = fig.add_subplot(111, frame_on=False)
    legend_ax_right = fig.add_subplot(111, frame_on=False)

    # Turn off the axis for both legends
    legend_ax_left.axis('off')
    legend_ax_right.axis('off')

    # Add legends to the invisible axes
    legend_ax_left.legend(handles_left+handles_left2, labels_left+labels_left2, loc='upper center', ncol=1, fontsize=fontsize, bbox_to_anchor=(0.25, 1.18))
    legend_ax_right.legend(handles_right, labels_right, loc='upper center', ncol=1, fontsize=fontsize, bbox_to_anchor=(0.75, 1.18))


    # Create a single shared legend
    fig.tight_layout() #rect=[0, 0, 1, 0.94])  # Adjust rect to make space for the legend: 
    # fig.legend(handles, _labels, loc='upper right', ncol=2, fontsize=fontsize+2, bbox_to_anchor=(0.925, 1.02))
    # Adjust layout and save
    fig.savefig(output_dir + "SplinesLikelihoodRatios.pdf")
    plt.close(fig)

    exit()



    ##### for JE: plot likelihood ratio as well....
    include_ratio = True
    if include_ratio:
        counts_ttH = hist_ttH.to_numpy()[0] / hist_ttH.sum().value
        counts_tHq = hist_tHq.to_numpy()[0] / hist_tHq.sum().value
        spline = model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy()

        bins = hist_ttH.to_numpy()[1]
        bin_centers = [(bins[ib] + bins[ib+1])/2 for ib in range(len(bins)-1)]

        ratio = np.log(counts_tHq / counts_ttH)

        fig2, ax = plt.subplots()
        ax.plot(model.acts[l][:, i][rank].cpu().detach().numpy(), 
            model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), 
            color="tab:red", lw=5, label="Spline", linestyle="dotted"
        )
        ax2 = ax.twinx()
        ax2.plot(
            bin_centers, 
            ratio, 
            color="tab:grey",
            lw=5,
            label="log(L(tH) / L(ttH))",
            linestyle="dotted",
            marker="x",
            # linestyle="None"
            )
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)
        ax.set_xlabel(fea)
        ax.set_ylabel("KAN-learned spline")
        ax2.set_ylabel("Log. likelihood ratio")
        fig.tight_layout()
        os.makedirs(output_dir + f"PlotsScaledFeaturesWithLH/", exist_ok=True)
        plt.savefig(output_dir + f"PlotsScaledFeaturesWithLH/{fea}.pdf")
    plt.close()

