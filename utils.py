import json
import numpy as np
import matplotlib.pyplot as plt
import torch 
import os

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
        

def load_model(model_path):
    return torch.load(model_path)

def load_thresholds(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata['thresholds']



def get_class_weights(dataset):
    # Initialize a counter for each class
    class_counts = np.zeros(dataset.num_classes)
    
    # Count each class occurrence
    for _, _, binary_label_vector in dataset.samples:
        class_counts += binary_label_vector
    
    # Compute class weights using log-weighting, with +1 normalization to avoid division by zero and log(0)
    class_weights = np.log((len(dataset) / (class_counts + 1)) + 1)
    
    # return tensor
    return torch.tensor(class_weights, dtype=torch.float)


def get_num_classes(root_dir):
    return len([f for f in os.listdir(root_dir) if f.endswith('_filtered.h5')])

import numpy as np

def custom_mcc(true_labels, preds):
    """
    Calculate the Matthews Correlation Coefficient for binary labels.
    
    Args:
    - true_labels (np.array): Numpy array of true binary labels.
    - preds (np.array): Numpy array of predicted binary labels.
    
    Returns:
    - mcc (float): The Matthews Correlation Coefficient.
    """
    
    # Calculate TP, TN, FP, FN
    TP = np.sum((preds == 1) & (true_labels == 1))
    TN = np.sum((preds == 0) & (true_labels == 0))
    FP = np.sum((preds == 1) & (true_labels == 0))
    FN = np.sum((preds == 0) & (true_labels == 1))
    
    # Calculate MCC
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if denominator == 0:
        return 0  # Return 0 or an appropriate value if the denominator is 0
    mcc = (TP * TN - FP * FN) / denominator
    
    return mcc




def plot_trial_logs(trial_logs):
    # Plotting validation losses for each trial
    plt.figure(figsize=(10, 6))
    for log in trial_logs:
        plt.plot(log['val_losses'], label=f"Trial {log['trial']} Loss")
    
    plt.title("Validation Losses per Trial")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot final validation loss for each set of hyperparameters
    plt.figure(figsize=(10, 6))
    val_losses = [log['final_val_loss'] for log in trial_logs]
    trials = [log['trial'] for log in trial_logs]
    plt.scatter(trials, val_losses)
    plt.title("Final Validation Loss per Trial")
    plt.xlabel("Trial")
    plt.ylabel("Final Validation Loss")
    plt.xticks(trials)
    plt.grid(True)
    plt.show()
