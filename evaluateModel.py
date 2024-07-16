import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
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

# Dictionary with paths to separate trainings
model_dirs = {
    "MLP 128-64-32-16": "./TrainingOutputs/MLP/LowAndHighLevels/128-64-32-16/",
    "MLP 1x32": "./TrainingOutputs/MLP/LowAndHighLevels/32/",
    "MLP 1x8": "./TrainingOutputs/MLP/LowAndHighLevels/8/",
#     "KAN 25-10-5-1": "./KAN_trainingsLowLevels/AllFeatures_25_10_5_1",
    "KAN 22-1": "./TrainingOutputs/KAN/LBFGS/LowAndHighLevels/22-1-LR1e-3/",
    "KAN 22-44-1": "./TrainingOutputs/KAN/LBFGS/LowAndHighLevels/22-44-1/",
    "KAN 22-10-5-2-1": "./TrainingOutputs/KAN/LBFGS/LowAndHighLevels/22-10-5-2-1/",
    "KAN 22-1 g=50": "./TrainingOutputs/KAN/LBFGS/LowAndHighLevels/22-1-grid50/",
    "KAN 22-10-5-2-1 Adam": "./TrainingOutputs/KAN/LBFGS/LowAndHighLevels/22-10-5-2-1_Adam/",
}
KAN_models_dict = {
    "KAN 22-1": KAN(width=[22, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 22-44-1": KAN(width=[22, 44, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 22-1 g=50": KAN(width=[22, 1], grid=50, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 22-10-5-2-1 Adam": KAN(width=[22, 10, 5, 2, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 22-10-5-2-1": KAN(width=[22, 10, 5, 2, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
}

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


# Function to create histograms
def create_histogram(predictions, weights, bins=20, range=(0, 1)):
    hist = Hist.new.Reg(bins, *range, name="output").Weight()
    hist.fill(predictions, weight=weights)
    return hist

# Function to calculate ROC AUC with bootstrapping
def bootstrap_roc_auc_rejection(predictions_dict, ttH_weights, tHq_weights, n_bootstrap=100, eff_for_rejection=0.8):
    aucs = []
    rejections = []
    y_true = np.concatenate([np.zeros(len(predictions_dict['ttH'])), np.ones(len(predictions_dict['tHq']))])
    y_score = np.concatenate([predictions_dict['ttH'], predictions_dict['tHq']])
    sample_weight = np.concatenate([ttH_weights, tHq_weights])
    # i = 0
    for _ in range(n_bootstrap):
        # i += 1
        # print(i)
        # Resample with replacement
        indices = resample(np.arange(len(y_true)), replace=True)
        y_true_resampled = y_true[indices]
        y_score_resampled = y_score[indices]
        sample_weight_resampled = sample_weight[indices]
        
        fpr, tpr, _ = roc_curve(y_true=y_true_resampled, y_score=y_score_resampled, sample_weight=sample_weight_resampled)
        roc_auc = integrate.trapz(tpr, fpr)
        aucs.append(roc_auc)
        rejection = find_rejection_for_tpr(fpr, tpr, eff_for_rejection)
        rejections.append(rejection)
    # print(f"{aucs=}")
    # print(np.mean(aucs), np.std(aucs))
    # exit()
    return np.mean(aucs), np.std(aucs), np.mean(rejections), np.std(rejections)

def find_rejection_for_tpr(fpr, tpr, target_tpr):
    # Find the index of the closest TPR value to the target TPR
    index = np.argmin(np.abs(tpr - target_tpr))
    # Get the corresponding FPR value
    fpr_value = fpr[index]
    # Calculate the rejection (1/FPR)
    rejection = 1 / fpr_value if fpr_value != 0 else 1
    return rejection


# Initialize figures for combined ROC curves
fig_roc_combined, ax_roc_combined = plt.subplots()
dict_AUC_rejection_nParams = {}

# Loop through each model directory
for model_label, model_dir in model_dirs.items():
    n_params = 0
    features_file = os.path.join(model_dir, "features_to_use.txt")
    with open(features_file, 'r') as f:
        features_to_use = [line.strip() for line in f]

    scaler_filename = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)

    if "KAN" not in model_dir:
        model_filename = os.path.join(model_dir, 'model.h5')
        model = tf.keras.models.load_model(model_filename)
        n_params = model.count_params()

        histograms = {}
        predictions_dict = {}
        for key, (df, weights) in test_data_dict.items():
            data_scaled = scaler.transform(df[features_to_use])
            predictions = model.predict(data_scaled, batch_size=1024).flatten()
            histograms[key] = create_histogram(predictions, weights)
            predictions_dict[key] = predictions
    else:
        model_filename = os.path.join(model_dir, 'model.pt')
        model = KAN_models_dict[model_label]
        model.load_state_dict(torch.load(model_filename))

        for i in range(len(model.width)-1):
            n_nodes = int(model.width[i][0])
            n_nodes_next = int(model.width[i+1][0])
            n_params += n_nodes * n_nodes_next * (model.grid + model.k)

        histograms = {}
        predictions_dict = {}
        for key, (df, weights) in KAN_test_data_dict.items():
            data_scaled = scaler.transform(df[features_to_use])
            data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
            with torch.no_grad():
                output = model(data_tensor)
            predictions = torch.sigmoid(output).numpy().reshape(-1)
            histograms[key] = create_histogram(predictions, weights)
            predictions_dict[key] = predictions

    labels = [r"$t\bar{t}H$", r"tH"]
    process_histograms = [histograms['ttH'], histograms['tHq']]
    plotter.plot_histograms(
        hists=[hist for hist in process_histograms],
        process_labels=labels,
        output_filename=os.path.join(model_dir, f'output_distributions.pdf'),
        axis_labels=("Model output", "Density"),
        normalize=True
    )

    fpr, tpr, _ = roc_curve(
        y_true=np.concatenate([np.zeros(len(predictions_dict['ttH'])), np.ones(len(predictions_dict['tHq']))]),
        y_score=np.concatenate([predictions_dict['ttH'], predictions_dict['tHq']]),
        sample_weight=np.concatenate([ttH_weights, tHq_weights])
    )
    roc_auc = integrate.trapz(tpr, fpr)

    eff_for_rejection = 0.8
    rejection = find_rejection_for_tpr(fpr, tpr, eff_for_rejection)

    mean_auc, std_auc, mean_rejection, std_rejection = bootstrap_roc_auc_rejection(predictions_dict, ttH_weights, tHq_weights, n_bootstrap=100, eff_for_rejection=eff_for_rejection)

    dict_AUC_rejection_nParams[model_label] = np.array([mean_auc, std_auc, mean_rejection, std_rejection, n_params])
    n_params_human_readable = human_readable_params(n_params)

    print(
        f"{model_dir = }, {roc_auc = }, {mean_auc = }, {std_auc = }"
    )
    fig, ax = plt.subplots()
    plot_roc_curve(fpr, tpr, roc_auc, "", ax, n_params_human_readable)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="best")
    fig.savefig(os.path.join(model_dir, f'ROC_curve.pdf'))
    plt.close(fig)

    plot_roc_curve(fpr, tpr, roc_auc, model_label, ax_roc_combined, n_params_human_readable)

# Finalize and save combined ROC curves
ax_roc_combined.set_xlabel("False Positive Rate")
ax_roc_combined.set_ylabel("True Positive Rate")
ax_roc_combined.set_ylim(0.4, 1.01)
ax_roc_combined.legend(loc="best")
fig_roc_combined.tight_layout()
fig_roc_combined.savefig(os.path.join(output_dir, 'combined_ROC_curves.pdf'))
plt.close(fig_roc_combined)

# AUC vs. N(params) plot
fig_auc, ax_auc = plt.subplots()
fig_rej, ax_rej = plt.subplots()
i_MLP = 0
i_KAN = 0
i_KAN_others = 0
for model_key in dict_AUC_rejection_nParams.keys():
    auc, auc_std, rejection, rejection_std, n_params = dict_AUC_rejection_nParams[model_key]
    markers = ["X", "o", "P", "*", "^", "v", "<", ">", "."]
    if "MLP" in model_key:
        errorbar_auc = ax_auc.errorbar(n_params, auc, yerr=auc_std, color="tab:orange", fmt=markers[i_MLP], label=model_key, ms=13, elinewidth=3) # , alpha=0.7
        errorbar_rej = ax_rej.errorbar(n_params, rejection, yerr=rejection_std, color="tab:orange", fmt=markers[i_MLP], label=model_key, ms=13, elinewidth=3) # , alpha=0.7
        i_MLP += 1
    elif ("KAN" in model_key) and (not "g" in model_key) and (not "Adam" in model_key):
        errorbar_auc = ax_auc.errorbar(n_params, auc, yerr=auc_std, color="tab:blue", fmt=markers[i_KAN], label=model_key, ms=13, elinewidth=3) # , alpha=0.7
        errorbar_rej = ax_rej.errorbar(n_params, rejection, yerr=rejection_std, color="tab:blue", fmt=markers[i_KAN], label=model_key, ms=13, elinewidth=3) # , alpha=0.7
        i_KAN += 1
    else:
        errorbar_auc = ax_auc.errorbar(n_params, auc, yerr=auc_std, color="tab:red", fmt=markers[i_KAN_others], label=model_key, ms=13, elinewidth=3) # , alpha=0.7
        errorbar_rej = ax_rej.errorbar(n_params, rejection, yerr=rejection_std, color="tab:red", fmt=markers[i_KAN_others], label=model_key, ms=13, elinewidth=3) # , alpha=0.7
        i_KAN_others += 1

    # Customize markers' alpha
    for cap in errorbar_auc[1]:
        cap.set_alpha(1)  # Set markers' alpha to 1

    # Customize error bars' alpha and linestyle
    for line in errorbar_auc[2]:
        # line.set_alpha(0.7)  # Set error bars' alpha
        line.set_linestyle('--')  # Set error bars' linestyle to dashed
    # Customize markers' alpha
    for cap in errorbar_rej[1]:
        cap.set_alpha(1)  # Set markers' alpha to 1

    # Customize error bars' alpha and linestyle
    for line in errorbar_rej[2]:
        # line.set_alpha(0.7)  # Set error bars' alpha
        line.set_linestyle('--')  # Set error bars' linestyle to dashed

# perhaps need to remove error bars from legend, now it shows small ones hidden by markers
ax_auc.legend(ncols=3, fontsize=16, columnspacing=0.5)
ax_auc.set_xlabel("Number of trainable parameters")
ax_auc.set_ylabel("ROC AUC")
# ax_auc.set_ylim(ax_auc.get_ylim()[0], 1.15*ax_auc.get_ylim()[1])
ax_auc.set_ylim(0.9, 0.955)
fig_auc.tight_layout()
fig_auc.savefig(os.path.join(output_dir, 'AUC_vs_N.pdf'))
ax_auc.set_xscale("log")
fig_auc.tight_layout()
fig_auc.savefig(os.path.join(output_dir, 'AUC_vs_N_log.pdf'))
plt.close(fig_auc)

ax_rej.legend(ncols=3, fontsize=16, columnspacing=0.5)
ax_rej.set_xlabel("Number of trainable parameters")
ax_rej.set_ylabel(r"$t\bar{t}H$ rejection $\left(\epsilon_{tH}=0.80\right)$")
# ax_rej.set_ylim(ax_rej.get_ylim()[0], 1.15*ax_rej.get_ylim()[1])
ax_rej.set_ylim(0, 1.15*ax_rej.get_ylim()[1])
fig_rej.tight_layout()
fig_rej.savefig(os.path.join(output_dir, 'rej_vs_N.pdf'))
ax_rej.set_xscale("log")
fig_rej.tight_layout()
fig_rej.savefig(os.path.join(output_dir, 'rej_vs_N_log.pdf'))
plt.close(fig_rej)

