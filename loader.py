import os
import torch
import json
import numpy as np

class ModelLoader:
    def __init__(self, load_path):
        """
        Initialize the ModelLoader with a path from where to load the model and metadata.
        :param load_path: Path where the model and metadata are saved.
        """
        self.load_path = load_path

    def load_model(self, filename='model.pth'):
        """
        Loads a PyTorch model from the specified file.
        :param filename: Filename from which to load the model.
        :return: Loaded PyTorch model.
        """
        model_path = os.path.join(self.load_path, filename)
        model = torch.load(model_path)
        return model

    def load_thresholds(self, filename='thresholds.npy'):
        """
        Loads thresholds from a specified numpy file.
        :param filename: Filename from which to load the thresholds.
        :return: Loaded thresholds as a numpy array.
        """
        thresholds_path = os.path.join(self.load_path, filename)
        thresholds = np.load(thresholds_path)
        return thresholds

    def load_losses(self, filename):
        """
        Loads losses from a specified numpy file.
        :param filename: Filename from which to load the losses.
        :return: Loaded losses as a numpy array.
        """
        loss_path = os.path.join(self.load_path, filename)
        losses = np.load(loss_path)
        return losses

    def load_metadata(self, filename='metadata.json'):
        """
        Loads metadata from the specified JSON file.
        :param filename: Filename from which to load the metadata.
        :return: Loaded metadata as a dictionary.
        """
        metadata_path = os.path.join(self.load_path, filename)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata

    def load(self):
        """
        Loads the model, thresholds, training and validation losses, and metadata.
        :return: A dictionary containing all loaded components.
        """
        model = self.load_model()
        thresholds = self.load_thresholds()
        train_losses = self.load_losses('train_losses.npy')
        val_losses = self.load_losses('val_losses.npy')
        metadata = self.load_metadata()

        return {
            'model': model,
            'thresholds': thresholds,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metadata': metadata
        }

# Example usage:
# load_path = 'path/to/save'
# model_loader = ModelLoader(load_path)
# loaded_components = model_loader.load()
# model = loaded_components['model']
# thresholds = loaded_components['thresholds']
# train_losses = loaded_components['train_losses']
# val_losses = loaded_components['val_losses']
# metadata = loaded_components['metadata']
