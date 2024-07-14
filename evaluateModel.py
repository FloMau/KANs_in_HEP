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
    # "lambda0": "./TrainingOutputs/DisCoTestsLam0",
#     "lambda10": "./TrainingOutputs/DisCoTestsLam10",
    # "lambda100Shallow512": "./TrainingOutputs/DisCoTestsLam100",
    # "lambda100Batch512": "./TrainingOutputs/DisCoLam100Deep512",
    # "lambda100Batch1024": "./TrainingOutputs/DisCoLam100Deep1024",
    # "lambda100Batch2048": "./TrainingOutputs/DisCoLam100Deep2048",
    # "lambda1000Batch4096": "./TrainingOutputs/DisCoLam1000Deep4096",
    # "lambda100Batch4096": "./TrainingOutputs/DisCoLam100Deep4096",
    # "lambda10Batch4096": "./TrainingOutputs/DisCoLam10Deep4096",
    # "lambda200Batch4096": "./TrainingOutputs/DisCoLam200Deep4096",
    # "lambda2000Batch4096": "./TrainingOutputs/DisCoLam2000Deep4096",
    "lambda100Batch8192": "./TrainingOutputs/DisCoLam100Deep8192",
    "lambda500Batch8192": "./TrainingOutputs/DisCoLam500Deep8192",
    # "lambda200": "./TrainingOutputs/DisCoTestsLam200",
    # "lambda1000": "./TrainingOutputs/DisCoTestsLam1000",
}

model_dirs = {
    "CP_even_only": "./TrainingOutputs/BenchmarkLam0Deep4096_CPevenOnlyLR1e-4/",
    "lambda0Batch4096": "./TrainingOutputs/DisCoPerClassUpdated/Lam0Deep4096",
    "lambda10Batch4096": "./TrainingOutputs/DisCoPerClassUpdated/Lam10Deep4096",
    "lambda20Batch4096": "./TrainingOutputs/DisCoPerClassUpdated/Lam20Deep4096",
}

model_dirs = {
    "CP even only": "./TrainingOutputs/BenchmarkLam0Deep4096_CPevenOnlyLR1e-4/",
    "lam_tH20_lam_ttH10": "./TrainingOutputs/DisCoPerClassIndividually/Lam_tH20_lam_ttH10Deep4096",
    "lam_tH50_lam_ttH10": "./TrainingOutputs/DisCoPerClassIndividually/Lam_tH50_lam_ttH10Deep4096",
    "lam_tH10_lam_ttH20": "./TrainingOutputs/DisCoPerClassIndividually/Lam_tH10_lam_ttH20Deep4096",
    "lam_tH10_lam_ttH50": "./TrainingOutputs/DisCoPerClassIndividually/Lam_tH10_lam_ttH50Deep4096",
}

model_dirs = {
    "CP even only": "./TrainingOutputs/BenchmarkLam0Deep4096_CPevenOnlyLR1e-4",
    "$\lambda = 0$": "./TrainingOutputs/DisCoTestsLam0",
    "$\lambda = 10$": "./TrainingOutputs/DisCoPerClassUpdated/Lam10Deep4096",
    "$\lambda = 20$": "./TrainingOutputs/DisCoPerClassUpdated/Lam20Deep4096",
    "$\lambda_{tH}=10$, $\lambda_{ttH}=50$": "./TrainingOutputs/DisCoPerClassIndividually/Lam_tH10_lam_ttH50Deep4096",
    #"$\lambda_{tH}=10$, $\lambda_{ttH}=10$ 8192": "./TrainingOutputs/DisCoPerClassIndividually/Lam_tH10_lam_ttH10Deep8192",
    "$\lambda_{tH}=10$, $\lambda_{ttH}=1$ 8192": "./TrainingOutputs/DisCoPerClassIndividually/Lam_tH10_lam_ttH1Deep8192",
    "$\lambda_{tH}=50$, $\lambda_{ttH}=10$ 8192": "./TrainingOutputs/DisCoPerClassIndividually/Lam_tH50_lam_ttH10Deep8192",
}

# model_dirs = {
#     "MLP 5x128": "./TrainingOutputs/BenchmarkLam0Deep4096_CPevenOnlyLR1e-4",
#     "MLP 1x32": "./TrainingOutputs/ShallowMLPs/32/",
#     "MLP 1x8": "./TrainingOutputs/ShallowMLPs/8/",
#     "KAN 25-10-5-1": "./KAN_trainingsLowLevels/AllFeatures_25_10_5_1",
#     "KAN 25-1": "./KAN_trainingsLowLevels/AllFeatures_25_1",
# }

model_dirs = {
#     "MLP 5x128": "./TrainingOutputs/BenchmarkLam0Deep4096_CPevenOnlyLR1e-4",
#     "MLP 1x32": "./TrainingOutputs/ShallowMLPs/32/",
    "MLP 1x8": "./TrainingOutputs/ShallowMLPs/8/",
    "MLP 1x8, $\lambda = 10$": "./TrainingOutputs/ShallowMLPs/8_Lam_tH10_lam_ttH10/",
    "MLP 1x32": "./TrainingOutputs/ShallowMLPs/8/",
    "MLP 1x32, $\lambda = 10$": "./TrainingOutputs/ShallowMLPs/32_Lam_tH10_lam_ttH10/",
#     "KAN 25-10-5-1": "./KAN_trainingsLowLevels/AllFeatures_25_10_5_1",
#     "KAN 25-1": "./KAN_trainingsLowLevels/AllFeatures_25_1",
}
KAN_models_dict = {
    "KAN 25-10-5-1": KAN(width=[25, 10, 5, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42),
    "KAN 25-1": KAN(width=[25, 1], grid=5, grid_range=(0,1), grid_eps=0, k=3, seed=42)
}

# output_dir = "./TrainingOutputs/CombinedLambdaScan/DisCoPerClass"
# output_dir = "./KANvsMLP/firstTests/"
output_dir = "./TrainingOutputs/ShallowMLPs/ROC_curves/"
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
df_ttHCPEven_test = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_test.csv'))
df_tHqCPEven_test = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_test.csv'))
df_ttHCPOdd_test = pd.read_csv(os.path.join(input_dir, 'ttHCPOdd_test.csv'))
df_tHqCPOdd_test = pd.read_csv(os.path.join(input_dir, 'tHqCPOdd_test.csv'))

# Optional: Read and append validation datasets
add_val = True
if add_val:
    df_ttHCPEven_val = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_val.csv'))
    df_tHqCPEven_val = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_val.csv'))
    df_ttHCPOdd_val = pd.read_csv(os.path.join(input_dir, 'ttHCPOdd_val.csv'))
    df_tHqCPOdd_val = pd.read_csv(os.path.join(input_dir, 'tHqCPOdd_val.csv'))

    df_ttHCPEven_test = pd.concat([df_ttHCPEven_test, df_ttHCPEven_val], ignore_index=True)
    df_tHqCPEven_test = pd.concat([df_tHqCPEven_test, df_tHqCPEven_val], ignore_index=True)
    df_ttHCPOdd_test = pd.concat([df_ttHCPOdd_test, df_ttHCPOdd_val], ignore_index=True)
    df_tHqCPOdd_test = pd.concat([df_tHqCPOdd_test, df_tHqCPOdd_val], ignore_index=True)

ttHCPEven_weights = df_ttHCPEven_test.pop("weights")
tHqCPEven_weights = df_tHqCPEven_test.pop("weights")
ttHCPOdd_weights = df_ttHCPOdd_test.pop("weights")
tHqCPOdd_weights = df_tHqCPOdd_test.pop("weights")

test_data_dict = {
    'ttHCPEven': [df_ttHCPEven_test, ttHCPEven_weights],
    'tHqCPEven': [df_tHqCPEven_test, tHqCPEven_weights],
    'ttHCPOdd': [df_ttHCPOdd_test, ttHCPOdd_weights],
    'tHqCPOdd': [df_tHqCPOdd_test, tHqCPOdd_weights],
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
fig_roc_even, ax_roc_even = plt.subplots()
fig_roc_odd, ax_roc_odd = plt.subplots()

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
        for n_nodes in model.width[:-1]:
            n_params += n_nodes * (model.grid + model.k)

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
    labels = ["ttH CP even", "tH CP even", "ttH CP odd", "tH CP odd"]
    process_histograms = [histograms['ttHCPEven'], histograms['tHqCPEven'], histograms['ttHCPOdd'], histograms['tHqCPOdd']]

    plotter.plot_histograms(
        hists=[hist for hist in process_histograms],
        process_labels=labels,
        output_filename=os.path.join(model_dir, f'output_distributions.pdf'),
        axis_labels=("Model output", "Density"),
        normalize=True
    )

    # Calculate and plot ROC curves

    # Calculate and plot ROC curves
    fpr_even, tpr_even, _ = roc_curve(
        y_true=np.concatenate([np.zeros(len(predictions_dict['ttHCPEven'])), np.ones(len(predictions_dict['tHqCPEven']))]),
        y_score=np.concatenate([predictions_dict['ttHCPEven'], predictions_dict['tHqCPEven']]),
        sample_weight=np.concatenate([ttHCPEven_weights, tHqCPEven_weights])
    )
    roc_auc_even = integrate.trapz(tpr_even, fpr_even)

    fpr_odd, tpr_odd, _ = roc_curve(
        y_true=np.concatenate([np.zeros(len(predictions_dict['ttHCPOdd'])), np.ones(len(predictions_dict['tHqCPOdd']))]),
        y_score=np.concatenate([predictions_dict['ttHCPOdd'], predictions_dict['tHqCPOdd']]),
        sample_weight=np.concatenate([ttHCPOdd_weights, tHqCPOdd_weights])
    )
    roc_auc_odd = integrate.trapz(tpr_odd, fpr_odd)
    print(
        f"{model_dir = }, {roc_auc_even = }, {roc_auc_odd =}"
    )
    fig, ax = plt.subplots()
    plot_roc_curve(fpr_even, tpr_even, roc_auc_even, f"CP Even", ax, n_params)
    plot_roc_curve(fpr_odd, tpr_odd, roc_auc_odd, f"CP Odd", ax, n_params)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="best")
    fig.savefig(os.path.join(model_dir, f'ROC_curves.pdf'))
    plt.close(fig)

    # Add to combined ROC plots
    plot_roc_curve(fpr_even, tpr_even, roc_auc_even, model_label, ax_roc_even, n_params)
    plot_roc_curve(fpr_odd, tpr_odd, roc_auc_odd, model_label, ax_roc_odd, n_params)

# Finalize and save combined ROC curves
ax_roc_even.set_xlabel("False Positive Rate")
ax_roc_even.set_ylabel("True Positive Rate")
ax_roc_even.set_ylim(0.4, 1.01)
ax_roc_even.legend(loc="best")
fig_roc_even.tight_layout()
fig_roc_even.savefig(os.path.join(output_dir, 'combined_roc_curves_cp_even.pdf'))

ax_roc_odd.set_xlabel("False Positive Rate")
ax_roc_odd.set_ylabel("True Positive Rate")
ax_roc_odd.set_ylim(0.4, 1.01)
ax_roc_odd.legend(loc="best")
fig_roc_odd.tight_layout()
fig_roc_odd.savefig(os.path.join(output_dir, 'combined_roc_curves_cp_odd.pdf'))

plt.close(fig_roc_even)
plt.close(fig_roc_odd)
