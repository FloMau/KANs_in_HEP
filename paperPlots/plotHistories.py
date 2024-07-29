import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.ROOT])

# Function to load history from CSV
def load_history(path):
    return pd.read_csv(path)

# Paths to your history files
paths = [
    '../TrainingOutputs/KANs26Jul2024/22-1-Adam/training_history.csv',
    '../TrainingOutputs/KANs26Jul2024/22-44-1-Adam/training_history.csv',
    '../TrainingOutputs/KANs26Jul2024/22-10-5-2-1-Adam/training_history.csv',
]

# Load histories
histories = [load_history(path) for path in paths]

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey='row')
font_size = 30
line_width = 4
handles, labels = [], []

for i, history in enumerate(histories):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    ax = axes[0, i]
    h1, = ax.plot(epochs, np.array(history['train_loss'])**2, label='Training Loss', linewidth=line_width)
    h2, = ax.plot(epochs, np.array(history['val_loss'])**2, label='Validation Loss', linewidth=line_width)
    ax.set_xlabel('', fontsize=font_size)
    # ax.set_yscale("log")
    ax.set_ylim(0.3, 0.6)
    ax.tick_params(axis='x', which='both', labelbottom=False)
    ax.tick_params(axis='both', labelsize=font_size-5)
    if i == 0:
        ax.set_ylabel('Cross-entropy', fontsize=font_size)
        handles.extend([h1, h2])
        labels.extend(['Training', 'Validation'])
    
    # Plot Accuracy
    ax = axes[1, i]
    h3, = ax.plot(epochs, 100*history['train_accuracy'], label='Training Accuracy', linewidth=line_width)
    h4, = ax.plot(epochs, 100*history['val_accuracy'], label='Validation Accuracy', linewidth=line_width)
    ax.set_xlabel('Training epoch', fontsize=font_size)
    ax.set_ylim(78, 85)
    ax.tick_params(axis='both', labelsize=font_size-5)
    if i == 0:
        ax.set_ylabel('Accuracy [%]', fontsize=font_size)
    #     handles.extend([h3, h4])
    #     labels.extend(['Training Accuracy', 'Validation Accuracy'])
# Set titles for each column
titles = ['KAN 22-1', 'KAN 22-44-1', 'KAN 22-10-5-2-1']
for i, ax in enumerate(axes[0]):
    ax.set_title(titles[i], fontsize=font_size+2)
# Adjust layout and create shared legend
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for the legend
fig.legend(handles, labels, loc='upper right', ncol=2, fontsize=font_size+2, bbox_to_anchor=(0.94, 1.0))

# Save the figure
output_dir = "./historyPlot"
os.makedirs(output_dir, exist_ok=True)
plot_filename = os.path.join(output_dir, 'CombinedTrainingHistory.pdf')
fig.savefig(plot_filename, bbox_inches='tight')
plt.close(fig)
