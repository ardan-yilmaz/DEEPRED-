from data_loader import get_data_loaders, GOAnnotationsDataset
from search_hyperparams import HyperparameterSearch
from saver import ModelSaver


class ModelTrainingOrchestrator:
    def __init__(self, dataset_path, num_classes, root_dir):
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.root_dir = root_dir

    def run(self):
        # Initialize and prepare your dataset
        
        # 1. Run hyperparameter search
        
        # 2. Run the threshold finder

        # 3. Save the model, train/val losses, thresholds, and evaluation metrics


