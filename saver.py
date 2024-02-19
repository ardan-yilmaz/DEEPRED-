import os
import torch
import json
import numpy as np

from utils import NumpyEncoder

def model_to_dict(model):
    """Converts a PyTorch model to a dictionary describing its architecture."""
    model_dict = {}
    for i, layer in enumerate(model.children()):
        layer_str = str(layer)
        if len(layer_str) > 100:  # Truncate long descriptions
            layer_str = layer_str[:100] + '...'
        model_dict[f'layer_{i}'] = layer_str
    return model_dict


class ModelSaver:
    def __init__(self, save_path):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def save_model(self, model, filename='model.pth'):
        model_path = os.path.join(self.save_path, filename)
        torch.save(model, model_path)  # Saving the entire model object

    def save_thresholds(self, thresholds, filename='thresholds.npy'):
        thresholds_path = os.path.join(self.save_path, filename)
        np.save(thresholds_path, thresholds)

    def save_losses(self, losses, filename):
        loss_path = os.path.join(self.save_path, filename)
        np.save(loss_path, losses)

    def save_metadata(self, config, model, eval_metrics, filename='metadata.json'):
        metadata_path = os.path.join(self.save_path, filename)
        metadata = {
            'hyperparams': config,
            'model_config': str(model),
            'evaluation_metrics': eval_metrics
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, cls=NumpyEncoder)

    def save(self, model, thresholds, config, eval_metrics, train_losses, val_losses):
        self.save_model(model)
        self.save_thresholds(thresholds)
        self.save_losses(train_losses, 'train_losses.npy')
        self.save_losses(val_losses, 'val_losses.npy')
        self.save_metadata(config, model, eval_metrics)

# Example usage:
# model_saver = ModelSaver('path/to/save')
# model_saver.save(model, thresholds, config, eval_metrics, train_losses, val_losses)
