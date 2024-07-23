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

model_dir = "../TrainingOutputs/KANs18Jul2024/22-1-LBFGS/"
output_dir = "./KAN_plots/"

# Load the configuration file
config_path = model_dir + "config.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Directory where the datasets are stored
input_dir = "../ML_datasets"

# Read datasets
stop = 10_000
df_ttH_train = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_train.csv'), nrows=stop)
df_tHq_train = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_train.csv'), nrows=stop)

train_data = pd.concat([df_ttH_train, df_tHq_train])
# Labels: 0 for ttHCPEven, 1 for tHqCPEven
train_labels = np.concatenate([np.zeros(len(df_ttH_train)), np.ones(len(df_tHq_train))])

for col in ["phoLeadPt", "phoSubleadPt", 'met', 'leptonPt', 'topMt', "ptH", 'bJetPt', 'jetLeadPt']:
    train_data[col] = np.log(1 + train_data[col].values)

train_data, train_labels = shuffle(train_data, train_labels, random_state=0)

train_weights = train_data.pop("weights")

# # Load the configuration from the JSON file
# config_path = 'configs/KANs/highAndLowLevels/22-1-testsOnly.json'
# with open(config_path, 'r') as config_file:
#     config = json.load(config_file)

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

# Save the list to a text file
features_file = os.path.join(output_dir, "features_to_use.txt")
with open(features_file, 'w') as f:
    for feature in features_to_use:
        f.write(f"{feature}\n")

print("INFO: training with features", features_to_use)
train_data = train_data[features_to_use]

short_names = plotter.tex_feature_names
features_labels = [short_names[f] if f in short_names.keys() else f for f in features_to_use]

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
# plt.figure(figsize=(10,6))
# # # model.plot(beta=100, in_vars=features, scale=10) #, out_vars=['y1', "y2"], title = 'My KAN', scale=0.04)
# KAN_plotter.plot(model, folder=output_dir+"Splines", beta=3, mask=False, metric='backward', scale=1, tick=True, sample=False, in_vars=feature_labels, out_vars=['ttH vs. tH'], title=None, varscale=0.2)
# # KAN_plotter.plot_custom(model, beta=50, in_vars=features_to_use, folder=output_dir+"Splines/", out_vars=['ttH vs. tH'], title = 'My KAN', scale=10, tick=True, tick_size=100)
# # # plt.tight_layout()
# plt.savefig(output_dir + "/KAN_modelFromEvalScript.pdf")
# plt.close()

dataset_ttH = train_data_scaled[train_labels == 0]
dataset_tHq = train_data_scaled[train_labels == 1]
weights_ttH = train_weights[train_labels == 0]
weights_tHq = train_weights[train_labels == 1]

plot_scaled_hists = True
if plot_scaled_hists:

    for i, fea in enumerate(features_to_use):

        hist_ttH = Hist.new.Reg(20, 0, 1).Weight()
        hist_tHq = Hist.new.Reg(20, 0, 1).Weight()
        hist_ttH.fill(
            dataset_ttH[:,i], weight=weights_ttH
        )
        hist_tHq.fill(
            dataset_tHq[:,i], weight=weights_tHq
        )

        fig, ax = plotter.plot_histograms(
            hists=[hist_ttH, hist_tHq],
            process_labels=["ttH", "tHq"],
            output_filename=output_dir + f"./PlotsScaledFeatures/{fea}.pdf",
            normalize=True,
            axis_labels=[fea, "Density"],
            include_flow=True,
            return_figure=True
        )
        ax2 = ax.twinx()

        depth = len(model.width) - 1

        l = 0
        _i = i
        # for _i in range(model.width_in[l]):
        for j in range(model.width_out[l+1]):
            print(l, _i, j)
            rank = torch.argsort(model.acts[l][:, i])
            num = rank.shape[0]

            #print(model.width_in[l])
            #print(model.width_out[l+1])
            symbolic_mask = model.symbolic_fun[l].mask[j][i]
            numeric_mask = model.act_fun[l].mask[i][j]

            plt.gca().patch.set_linewidth(1.5)
            # plt.axis('off')

            spline_line, = ax2.plot(model.acts[l][:, _i][rank].cpu().detach().numpy(),
                                    model.spline_postacts[l][:, j, _i][rank].cpu().detach().numpy(), color="tab:red", lw=5,
                                    label="Spline", linestyle="dotted")
        ax2.set_ylabel("Learned spline")
        # Ensure there is an upper margin
        # spline_y_max = model.spline_postacts[l][:, j, i].max().item()
        ylim_before = ax2.get_ylim()
        ax2.margins(y=0.2)
        ylim_after = ax2.get_ylim()
        ax2.set_ylim(ylim_before[0], ylim_after[1])
        # Add the spline to the legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + [spline_line], labels + ["Spline"])
        fig.tight_layout()
        os.makedirs(output_dir + f"PlotsScaledFeatures/", exist_ok=True)
        plt.savefig(output_dir + f"PlotsScaledFeatures/{fea}.pdf")
        plt.close()

        ##### for JE: plot likelihood ratio as well....
        include_ratio = False
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