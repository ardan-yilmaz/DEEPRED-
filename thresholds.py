import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from utils import custom_mcc

class ThresholdFinder:
    def __init__(self, model, num_classes, device='cpu'):
        """
        Initialize the ThresholdFinder with the trained model.
        :param model: Trained PyTorch model.
        :param num_classes: Number of classes in the dataset.
        :param device: Computation device ('cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes

    def predict_probabilities(self, loader):
        self.model.eval()
        probabilities_list = []
        true_labels_list = []

        with torch.no_grad():
            for _, inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = torch.sigmoid(self.model(inputs))
                probabilities_list.append(outputs.cpu().numpy())
                true_labels_list.append(labels.cpu().numpy())

        return np.concatenate(probabilities_list, axis=0), np.concatenate(true_labels_list, axis=0)


    def find_optimal_threshold(self, loader):
        """
        Finds the optimal threshold for each class based on MCC, applying a coarse search. 
        :param loader: DataLoader for the validation or test dataset.
        :return: Optimal threshold for each class.
        """
        probabilities, true_labels = self.predict_probabilities(loader)
        optimal_thresholds = np.zeros(self.num_classes)

        # Phase 1: Coarse Search
        coarse_start = 0.1
        coarse_end = 0.9
        coarse_step = 0.05
        coarse_ranges = np.zeros((self.num_classes, 2))  # To store the promising range for each class

        for i in range(self.num_classes):
            max_mcc = -1
            for threshold in np.arange(coarse_start, coarse_end + coarse_step, coarse_step):
                preds = (probabilities[:, i] > threshold).astype(int)
                mcc = custom_mcc(true_labels[:, i], preds)
                if mcc > max_mcc:
                    max_mcc = mcc
                    # Store the promising range around the current threshold
                    coarse_ranges[i] = [max(coarse_start, threshold - coarse_step),
                                        min(coarse_end, threshold + coarse_step)]

        # Phase 2: Fine Search within the identified promising range
        fine_step = 0.01
        for i in range(self.num_classes):
            max_mcc = -1
            fine_start, fine_end = coarse_ranges[i]
            for threshold in np.arange(fine_start, fine_end + fine_step, fine_step):
                preds = (probabilities[:, i] > threshold).astype(int)
                mcc = custom_mcc(true_labels[:, i], preds)
                if mcc > max_mcc:
                    max_mcc = mcc
                    optimal_thresholds[i] = threshold

        return optimal_thresholds

