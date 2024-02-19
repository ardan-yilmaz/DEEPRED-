import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef

class ModelEvaluator:
    def __init__(self, model, thresholds, device='cpu'):
        """
        Initialize the ModelEvaluator class with the trained model and optimal thresholds.
        :param model: Trained PyTorch model.
        :param thresholds: Optimal thresholds for each class determined by ThresholdFinder.
        :param device: Computation device ('cpu' or 'cuda').
        """
        self.model = model
        self.thresholds = thresholds
        self.device = device

    def evaluate(self, loader):
        """
        Evaluates the model on the given dataset with the specified thresholds.
        :param loader: DataLoader for the dataset.
        :return: Dictionary of evaluation metrics.
        """ 
        true_labels_list = []
        predictions_list = []

        self.model.eval()
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs, labels = inputs.to(self.device), targets.to(self.device)
                outputs = torch.sigmoid(self.model(inputs))

                preds = outputs > torch.Tensor(self.thresholds).to(self.device)
                
                true_labels_list.append(labels.cpu().numpy())
                predictions_list.append(preds.cpu().numpy())

        true_labels = np.vstack(true_labels_list)
        predictions = np.vstack(predictions_list)

        metrics = self.calculate_metrics(true_labels, predictions)
        return metrics

    @staticmethod
    def calculate_metrics(true_labels, predictions):
        """
        Calculates and returns the evaluation metrics.
        :param true_labels: Numpy array of true labels.
        :param predictions: Numpy array of predictions.
        :return: Dictionary of evaluation metrics including accuracy, precision, recall, F1-score, and MCC.
        """
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='micro')
        mcc = matthews_corrcoef(true_labels.ravel(), predictions.ravel())

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'F1-score': f1,
            'MCC': mcc
        }



# evaluator = ModelEvaluator(best_model, optimal_thresholds, device='cuda' if torch.cuda.is_available() else 'cpu')
# evaluation_metrics = evaluator.evaluate(eval_loader)

# print("Evaluation Metrics:", evaluation_metrics)
