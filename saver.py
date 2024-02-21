import os
import torch
import json
import numpy as np
import logging  # Import logging module

from utils import NumpyEncoder

# Configure basic logging
logging.basicConfig(level=logging.INFO, filename='model_saving_log.txt', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
        try:
            os.makedirs(save_path, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create directory {save_path}: {e}")

    def save_model(self, model, filename='model.pth'):
        model_path = os.path.join(self.save_path, filename)
        try:
            torch.save(model.state_dict(), model_path)  # It's a good practice to save model state dict
            logging.info(f"Model saved successfully at {model_path}")
        except Exception as e:
            logging.error(f"Failed to save model at {model_path}: {e}")

    def save_thresholds(self, thresholds, filename='thresholds.npy'):
        thresholds_path = os.path.join(self.save_path, filename)
        try:
            np.save(thresholds_path, thresholds)
            logging.info(f"Thresholds saved successfully at {thresholds_path}")
        except Exception as e:
            logging.error(f"Failed to save thresholds at {thresholds_path}: {e}")

    def save_losses(self, losses, filename):
        loss_path = os.path.join(self.save_path, filename)
        try:
            np.save(loss_path, losses)
            logging.info(f"Losses saved successfully at {loss_path}")
        except Exception as e:
            logging.error(f"Failed to save losses at {loss_path}: {e}")

    def save_metadata(self, config, model, eval_metrics, filename='metadata.json'):
        metadata_path = os.path.join(self.save_path, filename)
        try:
            metadata = {
                'hyperparams': config,
                'model_config': model_to_dict(model),  # Convert model to dict for more structured info
                'evaluation_metrics': eval_metrics
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, cls=NumpyEncoder)
            logging.info(f"Metadata saved successfully at {metadata_path}")
        except Exception as e:
            logging.error(f"Failed to save metadata at {metadata_path}: {e}")

    def save_testdata(self, dataset, filename='test_dataset.pth'):
        dataset_path = os.path.join(self.save_path, filename)
        try:
            torch.save(dataset, dataset_path)
            logging.info(f"Test dataset saved successfully at {dataset_path}")
        except Exception as e:
            logging.error(f"Failed to save test dataset at {dataset_path}: {e}")



    def save(self, model, thresholds, config, eval_metrics, train_losses, val_losses, best_model_test_dataset):
        try:
            self.save_model(model)
            self.save_thresholds(thresholds)
            self.save_losses(train_losses, 'train_losses.npy')
            self.save_losses(val_losses, 'val_losses.npy')
            self.save_metadata(config, model, eval_metrics)
            self.save_testdata(best_model_test_dataset)
            logging.info("All components saved successfully.")
        except Exception as e:
            logging.error(f"An error occurred during saving: {e}")
