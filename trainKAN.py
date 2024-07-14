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
from modules import plotter, KAN_plotter
from sklearn.utils import shuffle
import pickle
from hist import Hist
plt.style.use([hep.style.CMS])

# Directory where the datasets are stored
input_dir = "./ML_datasets"

# Read datasets
stop = 90_000
df_ttHCPEven_train = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_train.csv'), nrows=stop)
df_ttHCPEven_val = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_val.csv'), nrows=int(0.3*stop))
df_tHqCPEven_train = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_train.csv'), nrows=stop)
df_tHqCPEven_val = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_val.csv'), nrows=int(0.3*stop))

train_data = pd.concat([df_ttHCPEven_train, df_tHqCPEven_train])
val_data = pd.concat([df_ttHCPEven_val, df_tHqCPEven_val])
# Labels: 0 for ttHCPEven, 1 for tHqCPEven
train_labels = np.concatenate([np.zeros(len(df_ttHCPEven_train)), np.ones(len(df_tHqCPEven_train))])
val_labels = np.concatenate([np.zeros(len(df_ttHCPEven_val)), np.ones(len(df_tHqCPEven_val))])


for col in ["phoLeadPt", "phoSubleadPt", 'met', 'leptonPt', 'topMt', "ptH", 'bJetPt', 'jetLeadPt']:
    train_data[col] = np.log(1 + train_data[col].values)
    val_data[col] = np.log(1 + val_data[col].values)

train_data, train_labels = shuffle(train_data, train_labels, random_state=0)
val_data, val_labels = shuffle(val_data, val_labels, random_state=0)

train_weights = train_data.pop("weights")
val_weights = val_data.pop("weights")

features = train_data.columns

### all features currently in datasets
features_to_use = ['ptH', 'yH', 'phiH', 'nJets', 'nLeptons', 'nJetsCentral',
       'nBjetsCentral', 'nJetsForward', 'met', 'leptonPt', 'leptonEta',
       'leptonPhi', 'topMt', 'bJetPt', 'bJetEta', 'bJetPhi', 'jetLeadPt',
       'jetLeadEta', 'jetLeadPhi', 'delta_eta_b_j', 'rapiditySumJH',
       'rapidityDiffJH', 'DeltaY_Hb', 'DeltaY_Hl']

lam = 1e-3
output_dir = f"./TrainingOutputs/KAN/test/"
os.makedirs(output_dir, exist_ok=True)
# Save the list to a text file
features_file = os.path.join(output_dir, "features_to_use.txt")
with open(features_file, 'w') as f:
    for feature in features_to_use:
        f.write(f"{feature}\n")

train_data = train_data[features_to_use]
val_data = val_data[features_to_use]

# Feature columns
features = train_data.columns

short_names = {
'ptH': 'ptH', 
'yH': 'yH',
'nJets': 'nJ',
'nLeptons': "nL",
'nJetsCentral': "nJC",
'nBjetsCentral': "nBJ",
'nJetsForward': "nJF",
'met': "MET",
'leptonPt': "pTL",
'topMt': "mTt",
'delta_eta_b_j': "deBJ",
'rapiditySumJH': "yH+J",
'rapidityDiffJH': "dyHJ",
'DeltaY_Hb': "dyHb",
'DeltaY_Hl': "dyHL",
"jetLeadPt": "pTJ",
"jetLeadEta": "etaJ",
"jetLeadPhi": "phiJ",
"bJetPt": "pTB",
"bJetEta": "etaB",
"bJetPhi": "phiB",
"leptonPt": "pTL",
"leptonEta": "etaL",
"leptonPhi": "phiL",
}
features = [short_names[f] if f in short_names.keys() else f for f in features]
print("INFO: training with features", features)

# Apply standard scaling
scaler = StandardScaler()
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
model = KAN(width=[input_dim, 3, 3, 3, 1],
            grid=5,
            # grid_range=(0,1),
            # grid_eps=0.,
            k=3,
            seed=42)

# Save the model
continueTraining = False
if continueTraining:
    modelSaveName = "modelGridExt.pt"
    model_filename = os.path.join(output_dir, 'model.pt')
    model.load_state_dict(torch.load(model_filename))

    model = KAN(width=[12,3,3,3,1], grid=50, k=3).initialize_from_another_model(model, dataset['train_input'])
    # model.remove_edge(0,2,0)
    # model.remove_edge(0,5,0)
    # model.remove_edge(0,5,1)
    # model(dataset["train_input"])
    # for i in range(2):
    #     print("Trying to suggest ", i)
    #     try:
    #         model.suggest_symbolic(1,i,0)
    #     except:
    #         print("failed suggesting symbolic")
    # model.fix_symbolic(1,1,0,'x')

if not continueTraining:
    modelSaveName = 'model.pt' 

    # model.prune()

# Define custom BCE loss functions for training and testing
def BCE_with_sigmoid(predictions, targets):
    # Apply sigmoid to normalize the output
    sigmoid_predictions = torch.sigmoid(predictions)
    # Compute binary cross-entropy loss
    loss = F.binary_cross_entropy(sigmoid_predictions, targets)
    return loss

def train_acc():
    return torch.mean(   ((torch.sigmoid(model(dataset['train_input'])) >= 0.5).float() == dataset['train_label']).float()   )

def test_acc():
    return torch.mean(   ((torch.sigmoid(model(dataset['test_input'])) >= 0.5).float() == dataset['test_label']).float()   )
model.speed()
# Train the model using the KAN's internal training procedure with custom BCE loss and accuracy metrics
results = model.fit(
    dataset,
    opt="Adam", # "LBFGS"
    lr=1e-2,
    steps=250,
    batch=512,
    metrics=(train_acc, test_acc),
    loss_fn=BCE_with_sigmoid,
    lamb=lam,
    # update_grid=False
)

print(results)

# print(
#     model.act_fun[0].grid[0].data
# )

model_filename = os.path.join(output_dir, modelSaveName)
torch.save(model.state_dict(), model_filename)

plt.figure(figsize=(10,6))
# model.plot(beta=100, in_vars=features, scale=10) #, out_vars=['y1', "y2"], title = 'My KAN', scale=0.04)
KAN_plotter.plot_custom(model, beta=10, in_vars=features, out_vars=['ttH vs. tH'], title = 'My KAN', scale=10, tick_size=100, tick=True)
# plt.tight_layout()
if not continueTraining:
    plt.savefig(output_dir + "/KAN_model.pdf")
else:
    plt.savefig(output_dir + "/KAN_model2.pdf")
plt.close()
## prune and plot model
modelPruned = model.prune()
modelPruned(dataset['train_input'])
# # fig, ax = plt.subplots(figsize=(20,10))
plt.figure(figsize=(10,6))
# modelPruned.plot(beta=100, in_vars=features, scale=10) #, out_vars=['y1', "y2"], title = 'My KAN', scale=0.04)
KAN_plotter.plot_custom(modelPruned, beta=10, in_vars=features, out_vars=['ttH vs. tH'], title = 'My KAN', scale=10, tick_size=100)
# plt.tight_layout()
plt.savefig(output_dir + "/KAN_modelPruned.pdf")
plt.close()


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
ax[0].plot(epochs, results['train_loss'], label='Training Loss')
ax[0].plot(epochs, results['test_loss'], label='Validation Loss')
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
