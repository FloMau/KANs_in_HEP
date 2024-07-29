import sys
sys.path.append("./..")
from modules import plotter  # Assuming plotter is defined in your modules
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist
from modules import plotter  # Assuming plotter is defined in your modules

plt.style.use([hep.style.ROOT])

# Directory where the datasets are stored
input_dir = "../ML_datasets"

# Read datasets
df_ttH = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_train.csv'))
df_tHq = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_train.csv'))
df_ttH_2 = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_val.csv'))
df_tHq_2 = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_val.csv'))
df_ttH_3 = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_test.csv'))
df_tHq_3 = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_test.csv'))

df_ttH = pd.concat([df_ttH, df_ttH_2, df_ttH_3], ignore_index=True)
df_tHq = pd.concat([df_tHq, df_tHq_2, df_tHq_3], ignore_index=True)

weights_ttH = df_ttH["weights"]
weights_tHq = df_tHq["weights"]

# Features to use
features_to_use = [
    "ptH", "yH", 
    "jetLeadPt", "jetLeadEta",
    "topMt",
    "nJets", 
    "delta_eta_b_j", "rapiditySumJH", "DeltaY_Hb", "DeltaY_Hl"
]

tex_feature_names = plotter.tex_feature_names  # Dictionary of LaTeX formatted feature names
labels = [tex_feature_names.get(f, f) for f in features_to_use]

# Initialize histograms using plotter.get_dict_hists
hists_ttH = plotter.get_dict_hists()
hists_tHq = plotter.get_dict_hists()

# Fill histograms with the data
stop = None  # Set to limit number of entries for testing
for feature in features_to_use:

    hist_key =  feature.replace("jetLeadPt", "leadingJetPt").replace("jetLeadEta", "leadingJetEta")
    data_ttH = df_ttH[feature].values
    data_tHq = df_tHq[feature].values

    hists_ttH[hist_key][0].fill(data_ttH, weight=weights_ttH)
    hists_tHq[hist_key][0].fill(data_tHq, weight=weights_tHq)

# Plotting
fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey='row')
handles, _labels = [], []
for idx, feature in enumerate(features_to_use):
    hist_key =  feature.replace("jetLeadPt", "leadingJetPt").replace("jetLeadEta", "leadingJetEta")
    ax = axes[idx // 5, idx % 5]

    label = labels[idx]
    if "_T" in label: 
        label += "$\ $[GeV]"
    _, ax = plotter.plot_histograms(
        hists=[hists_ttH[hist_key][0], hists_tHq[hist_key][0]],
        process_labels=[r"$\mathit{t\bar{t}H}$", r"$\mathit{tH}$"],
        ax=ax,
        normalize=True,
        axis_labels=(label, 'Fraction of events' if idx % 5 == 0 else ''),
        include_flow=True,
        return_figure=True
    )
    # Collect the handles and labels for the shared legend
    handles, _labels = ax.get_legend_handles_labels()
    # handles.extend(h)
    # _labels.extend(l)
    # Remove the legend from the current subplot
    binning = hists_ttH[hist_key][0].to_numpy()[1]
    ax.set_xlim(binning[0], binning[-1])
    ax.get_legend().remove()

    # Set the linewidth for all lines in the current subplot
    # for line in ax.get_lines():
    #     line.set_linewidth(8)  # Set the desired linewidth here
    # Adjust linewidth for all plot elements
    for line in ax.get_lines():
        line.set_linewidth(4)
    for patch in ax.patches:
        patch.set_linewidth(4)
    for collection in ax.collections:
        collection.set_linewidth(4)


# axes.flatten()[0].set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])  # Replace (0, 1) with your desired ylim
axes.flatten()[0].set_ylim(axes.flatten()[0].get_ylim()[0], 0.94*axes.flatten()[0].get_ylim()[1]) 
axes.flatten()[5].set_ylim(axes.flatten()[5].get_ylim()[0], 0.71*axes.flatten()[5].get_ylim()[1]) 

fontsize = 30
for ax in axes.flatten():
    ax.xaxis.label.set_size(fontsize-2)  # Set the x-axis label fontsize
    ax.yaxis.label.set_size(fontsize)  # Set the y-axis label fontsize



# Create a single shared legend
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for the legend
fig.legend(handles, _labels, loc='upper right', ncol=2, fontsize=fontsize+2, bbox_to_anchor=(0.925, 1.02))
# Adjust layout and save

# Adjust layout and save
os.makedirs("./PlotsInputFeatures", exist_ok=True)
fig.savefig("./PlotsInputFeatures/inputFeatures.pdf")
plt.close(fig)
