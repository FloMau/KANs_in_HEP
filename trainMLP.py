import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use([hep.style.ROOT])
import json
import shutil
import os
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Train KAN models with specified configuration.')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')

# Parse the arguments
args = parser.parse_args()

# Load the configuration file
config_path = args.config
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Directory where the datasets are stored
input_dir = "./ML_datasets"

# Read datasets
stop = 100_000
stop_val = 33_000
df_ttHCPEven_train = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_train.csv'), nrows=stop)
df_ttHCPEven_val = pd.read_csv(os.path.join(input_dir, 'ttHCPEven_val.csv'), nrows=stop_val)
df_tHqCPEven_train = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_train.csv'), nrows=stop)
df_tHqCPEven_val = pd.read_csv(os.path.join(input_dir, 'tHqCPEven_val.csv'), nrows=stop_val)

train_weights_ttHCPEven = np.abs(df_ttHCPEven_train.pop("weights")).astype(np.float32)
train_weights_tHqCPEven = np.abs(df_tHqCPEven_train.pop("weights")).astype(np.float32)
val_weights_ttHCPEven = np.abs(df_ttHCPEven_val.pop("weights")).astype(np.float32)
val_weights_tHqCPEven = np.abs(df_tHqCPEven_val.pop("weights")).astype(np.float32)

train_data = pd.concat([df_ttHCPEven_train, df_tHqCPEven_train]).astype(np.float32)
val_data = pd.concat([df_ttHCPEven_val, df_tHqCPEven_val]).astype(np.float32)
# Labels: 0 for ttHCPEven, 1 for tHqCPEven
train_labels = np.concatenate([np.zeros(len(df_ttHCPEven_train)), np.ones(len(df_tHqCPEven_train))]).astype(np.float32)
val_labels = np.concatenate([np.zeros(len(df_ttHCPEven_val)), np.ones(len(df_tHqCPEven_val))]).astype(np.float32)

# normalise weights
train_weights_ttHCPEven = train_weights_ttHCPEven * len(train_weights_ttHCPEven) / np.sum(train_weights_ttHCPEven)
train_weights_tHqCPEven = train_weights_tHqCPEven * len(train_weights_tHqCPEven) / np.sum(train_weights_tHqCPEven)
val_weights_ttHCPEven = val_weights_ttHCPEven * len(val_weights_ttHCPEven) / np.sum(val_weights_ttHCPEven)
val_weights_tHqCPEven = val_weights_tHqCPEven * len(val_weights_tHqCPEven) / np.sum(val_weights_tHqCPEven)

train_weights = pd.concat([train_weights_ttHCPEven, train_weights_tHqCPEven]).astype(np.float32)
val_weights = pd.concat([val_weights_ttHCPEven, val_weights_tHqCPEven]).astype(np.float32)
## are 1.0 for all events....

train_data, train_labels, train_weights = shuffle(train_data, train_labels, train_weights, random_state=0)
val_data, val_labels, val_weights = shuffle(val_data, val_labels, val_weights, random_state=0)

print("INFO: number of samples for ttH and tH:", len(df_ttHCPEven_train), len(df_tHqCPEven_train))

# Extract settings from the config
features_to_use = config['features_to_use']
epochs = config['epochs']
patience = config['patience']
LR = config['learning_rate']
batch_size = config['batch_size']
num_nodes = config['num_nodes']
output_dir = config['output_dir']

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Copy the configuration file to the output directory
shutil.copy(config_path, os.path.join(output_dir, 'config.json'))

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
val_data = val_data[features_to_use] 

# Apply standard scaling
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
val_data_scaled = scaler.transform(val_data)


def build_model(input_dim, num_nodes=[64, 64], output_dim=1, activation='sigmoid'):
    model = Sequential()
    model.add(Dense(num_nodes[0], input_dim=input_dim, activation='relu'))
    for layer_nodes in num_nodes[1:]:
        model.add(Dense(layer_nodes, activation='relu'))
    model.add(Dense(output_dim, activation=activation))
    return model

# Save the scaler
scaler_filename = os.path.join(output_dir, 'scaler.pkl')
pd.to_pickle(scaler, scaler_filename)

# Parameters for the model
input_dim = train_data_scaled.shape[1]
output_dim = 1 # Binary classification

@tf.function
def BCE(y_true, y_pred, sample_weight):
    y_true = tf.reshape(y_true, tf.shape(y_pred))
    # Binary Cross-Entropy loss
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) * sample_weight
    return tf.reduce_mean(bce_loss)

# Instantiate and compile the custom model
model = build_model(input_dim, num_nodes, output_dim, "sigmoid")
print(model.summary())
optimizer = Adam(learning_rate=LR)
# Prepare the data as TensorFlow datasets

train_dataset = tf.data.Dataset.from_tensor_slices((train_data_scaled, train_labels, train_weights))
val_dataset = tf.data.Dataset.from_tensor_slices((val_data_scaled, val_labels, val_weights))

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

# DataFrame to store the loss
loss_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])

# Initialize history dictionary to store training and validation loss and accuracy
history = {
    'bce': [],
    'val_bce': [],
    'accuracy': [],
    'val_accuracy': []
}

train_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
val_accuracy_metric = tf.keras.metrics.BinaryAccuracy()

# Variables to track the best epoch and early stopping
best_metric = float('inf')
best_model_weights = None
best_epoch = 0
epochs_since_last_improvement = 0

# Training loop
for epoch in range(epochs):
    print("Starting epoch", epoch + 1)
    train_bce = 0.0
    train_accuracy = 0.0
    num_batches = 0
    for step, (x_batch_train, y_batch_train, w_batch_train) in enumerate(train_dataset):

        train_accuracy_metric.reset_state()
        val_accuracy_metric.reset_state()

        with tf.GradientTape() as tape:
            y_pred_train = model(x_batch_train, training=True)
            bce_loss = BCE(y_batch_train, y_pred_train, w_batch_train)
            disCoLoss = tf.constant(0., dtype=tf.float32)
            loss_value = bce_loss

        train_accuracy_metric.update_state(y_batch_train, y_pred_train)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_bce += bce_loss

        train_accuracy += train_accuracy_metric.result().numpy()
        num_batches += 1
    train_bce /= num_batches
    train_accuracy /= num_batches

    # Validation loop
    val_bce = 0.0
    val_accuracy = 0.0
    num_batches = 0
    for x_batch_val, y_batch_val, w_batch_val in val_dataset:
        y_pred_val = model(x_batch_val, training=False)
        bce_loss = BCE(y_batch_val, y_pred_val, w_batch_val)
        val_accuracy_metric.update_state(y_batch_val, y_pred_val)

        val_bce += bce_loss
        val_accuracy += val_accuracy_metric.result().numpy()
        num_batches += 1

    val_bce /= num_batches
    val_accuracy /= num_batches

    # Log the losses and accuracy
    history['bce'].append(train_bce.numpy())
    history['val_bce'].append(val_bce.numpy())
    history['accuracy'].append(train_accuracy)
    history['val_accuracy'].append(val_accuracy)
    print(f"Epoch {epoch + 1}/{epochs}, Train BCE: {train_bce.numpy()}, Train Accuracy: {train_accuracy}, Val BCE: {val_bce.numpy()}, Val Accuracy: {val_accuracy}")

    # Determine the current metric based on the condition
    current_metric = val_bce.numpy()

    if current_metric < best_metric:
        best_metric = current_metric
        best_epoch = epoch
        best_model_weights = model.get_weights()
        epochs_since_last_improvement = 0  # Reset the counter if there's improvement
    else:
        epochs_since_last_improvement += 1

    # Check if we should stop early
    if epochs_since_last_improvement >= patience:
        print(f"No improvement for {patience} consecutive epochs, stopping early.")
        break

# Save the best model weights
model.set_weights(best_model_weights)
# Save the model
model_filename = os.path.join(output_dir, 'model.h5')
print(f"INFO: saving model from epoch {best_epoch}")
model.save(model_filename)

# Save the history DataFrame to a CSV file
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)


# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
font_size = 20

# Loss plot with secondary y-axis for DisCo
color_bce = 'tab:blue'
color_disco = 'tab:orange'

ax0 = ax[0]
ax0.plot(history['bce'], label='Training BCE', color=color_bce)
ax0.plot(history['val_bce'], label='Validation BCE', linestyle='dashed', color=color_bce)
ax0.set_xlabel('Training epoch', fontsize=font_size)
ax0.set_ylabel('Binary Cross-Entropy', fontsize=font_size, color=color_bce)
ax0.tick_params(axis='y', labelcolor=color_bce)
ax0.legend(loc='upper left', fontsize=0.65*font_size)

# Accuracy plot
ax[1].plot(history['accuracy'], label='Training Accuracy', color='tab:blue')
ax[1].plot(history['val_accuracy'], label='Validation Accuracy', linestyle='dashed', color='tab:orange')
ax[1].set_xlabel('Training epoch', fontsize=font_size)
ax[1].set_ylabel('Accuracy', fontsize=font_size)
ax[1].set_ylim(0.6, 1.0)
ax[1].legend(loc='upper left', fontsize=0.8*font_size)

# Save plot
plot_filename = os.path.join(output_dir, 'TrainingHistory.pdf')
fig.tight_layout()
fig.savefig(plot_filename)

plt.show()

