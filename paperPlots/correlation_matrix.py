import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from kan import KAN
import sys
import glob
sys.path.append("./..")
from modules import plotter
from hist import Hist
plt.style.use([hep.style.ROOT])


# Directory where the datasets are stored
input_dir = "../ML_datasets"

# Read datasets
df_ttH = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_train.csv'))
df_tHq = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_train.csv'))

df_combined = pd.concat([df_ttH, df_tHq])

features_to_use = [
    "ptH", "yH", "phiH",
    "bJetPt", "bJetEta", "bJetPhi",
    "leptonPt", "leptonEta", "leptonPhi",
    "met", "metPhi",
    "jetLeadPt", "jetLeadEta", "jetLeadPhi",
    "nJets","nLeptons", "nBjetsCentral",
    "topMt",
    "delta_eta_b_j", "rapiditySumJH", "DeltaY_Hb", "DeltaY_Hl"
]

short_names = {
'ptH': r'$\mathit{p}_{\mathrm{T}}(\mathit{\gamma\gamma})$', # r'$p_{T}(\gamma\gamma)$', 
'yH': '$\mathit{y}(\mathit{\gamma\gamma})$',
'phiH': '$\mathit{\phi}(\mathit{\gamma\gamma})$',
'nJets': '$\mathit{N}$(jets)',
'nLeptons': "$\mathit{N}$(leptons)",
'nJetsCentral': "nJC",
'nBjetsCentral': "$\mathit{N}$($\mathit{b}$-jets)",
'nJetsForward': "nJF",
'met': "$\mathit{E}_T^{miss}$",
'metPhi': r"$\mathit{\phi}\left(\mathit{E}_T^{miss}\right)$",
'topMt': "$\mathit{m}_T(\mathit{t})$",
'delta_eta_b_j': "$\Delta\mathit{\eta(b, j)}$",
'rapiditySumJH': "$\mathit{y}^{j} \oplus y^{\mathit{\gamma\gamma}}$",
'rapidityDiffJH': "$\Delta \mathit{y}(\mathit{j}, \mathit{\gamma\gamma})$",
'DeltaY_Hb': "$\Delta \mathit{y}(\mathit{\gamma\gamma}, b)$",
'DeltaY_Hl': "$\Delta \mathit{y}(\mathit{\gamma\gamma}, \mathit{\ell})$",
"jetLeadPt": "$\mathit{p}_T(\mathit{j})$",
"jetLeadEta": "$\mathit{\eta}(\mathit{j})$",
"jetLeadPhi": "$\mathit{\phi}(\mathit{j})$",
"bJetPt": "$\mathit{p}_T(\mathit{b})$",
"bJetEta": "$\mathit{\eta}(\mathit{b})$",
"bJetPhi": "$\mathit{\phi}(\mathit{b})$",
"leptonPt": "$\mathit{p}_T(\mathit{\ell})$",
"leptonEta": "$\mathit{\eta}(\mathit{\ell})$",
"leptonPhi": "$\mathit{\phi}(\mathit{\ell})$",
}

labels = [short_names[f] if f in short_names.keys() else f for f in features_to_use]




df_combined_ttH = df_ttH[features_to_use]
df_combined_tHq = df_tHq[features_to_use]

# Calculate the correlation matrices
corr_matrix_ttH = df_combined_ttH.corr() * 100
corr_matrix_tHq = df_combined_tHq.corr() * 100

# Combine matrices with NaNs in the diagonal
combined_corr_matrix = corr_matrix_ttH.copy()
for i in range(len(combined_corr_matrix)):
    for j in range(len(combined_corr_matrix)):
        if i > j:
            combined_corr_matrix.iloc[i, j] = corr_matrix_ttH.iloc[i, j]
        elif i < j:
            combined_corr_matrix.iloc[i, j] = corr_matrix_tHq.iloc[i, j]
        # else:
        #     combined_corr_matrix.iloc[i, j] = np.nan

# Plot the correlation matrix
fig, ax = plt.subplots(figsize=(12, 10))
cax = ax.matshow(combined_corr_matrix, cmap='coolwarm')

# Add color bar
cbar = fig.colorbar(cax)
cbar.set_label('Correlation coefficient [%]') #, fontsize=14

# Set axis labels
ax.set_xticks(np.arange(len(combined_corr_matrix.columns)))
ax.set_yticks(np.arange(len(combined_corr_matrix.columns)))
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

# Rotate the x tick labels
plt.setp(ax.get_xticklabels(), rotation=60, ha="left", rotation_mode="anchor")

# Remove x/y tick markers
ax.tick_params(axis="both", which="both", length=0)

# Print the correlation values for ttH (lower triangle) and tHq (upper triangle)
for i in range(len(corr_matrix_ttH.columns)):
    for j in range(len(corr_matrix_ttH.columns)):
        if i > j:
            corr_ttH = corr_matrix_ttH.iloc[i, j]
            if abs(corr_ttH) >= 10:
                text = ax.text(j, i, f"{corr_ttH:.0f}", ha="center", va="center", color="black", fontsize=16)
        elif i < j:
            corr_tHq = corr_matrix_tHq.iloc[i, j]
            if abs(corr_tHq) >= 10:
                text = ax.text(j, i, f"{corr_tHq:.0f}", ha="center", va="center", color="black", fontsize=16)

ax.text(15, 6.5, r"$\boldsymbol{tH}$", color="black",fontsize=34)
ax.text(6.5, 15, r"$\boldsymbol{t\bar\boldsymbol{t} H}$", color="black",fontsize=34)
# Adjust layout and save
fig.tight_layout()
fig.savefig("./correlationMatrix.pdf")
plt.close()