from data_loader import get_data_loaders
from MLP_classifier import MLPClassifier
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy
import random 
import torch.nn as nn


class HyperparameterSearch:
    def __init__(self, dataset, num_classes, class_weights, num_trials=25, max_epochs=250, device='cpu'):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_trials = num_trials
        self.max_epochs = max_epochs
        self.device = device
        self.class_weights = class_weights.to(self.device)


    def generate_hidden_layer_sizes(self):
        # Decide on the number of layers, between 2 and 5
        num_layers = random.randint(2, 5)
        # Dynamically adjust the size of the layers based on num_classes
        last_layer_size = max(16, self.num_classes * 2)  # Ensure minimum size for the last layer
        hidden_layers = [last_layer_size]

        for _ in range(1, num_layers):
            # Increase layer size for each previous layer, ensuring a minimum value
            next_layer_size = hidden_layers[-1] * 2
            hidden_layers.append(next_layer_size)

        # Reverse to have largest layers first, decreasing towards the output
        hidden_layers.reverse()
        return hidden_layers
    

    def get_random_params(self):
        return {
            'lr': np.random.uniform(1e-5, 1e-1),
            'dropout_rate': np.random.uniform(0.0, 0.5),
            'batch_size': np.random.choice([16, 32, 64, 128]),
            'hidden_layers': self.generate_hidden_layer_sizes(),
        }


    def train_model(self, model, patience, train_loader, val_loader, lr):
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr)
        best_val_loss = float('inf')
        best_model_state = None
        train_losses = []
        val_losses = []
        epochs_no_improve = 0

        for _ in range(self.max_epochs):
            model.train()
            total_train_loss = 0.0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Adjust validation loss calculation to match training
            total_val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for _, inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break  # Early stopping

        return train_losses, val_losses, best_model_state

    def run_search(self):
        best_val_loss = float('inf')
        best_model_state = None
        best_train_losses = []
        best_val_losses = []
        best_trial_params = None
        best_model_test_loader = None # kept for thresholding
        best_model_test_dataset = None # kept for thresholding

        for _ in range(self.num_trials):
            trial_params = self.get_random_params()
            model = MLPClassifier(hidden_layers=trial_params['hidden_layers'], dropout_rate=trial_params['dropout_rate'], num_classes=self.num_classes).to(self.device)
            train_loader, val_loader, test_loader, test_dataset = get_data_loaders(initial_dataset = self.dataset, batch_size=int(trial_params['batch_size']))
            train_losses, val_losses, model_state = self.train_model(model, patience=10, train_loader=train_loader, val_loader=val_loader, lr=trial_params['lr'])

            # if the min loss is better than the best val loss, update the best val loss and model state
            final_val_loss = np.min(val_losses)
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_model_state = model_state
                best_train_losses = train_losses
                best_val_losses = val_losses
                best_trial_params = trial_params
                best_model_test_loader = test_loader
                best_model_test_dataset = test_dataset

        

        # After search, load the best model state
        try:
            best_model = MLPClassifier(hidden_layers=best_trial_params['hidden_layers'],
                                    dropout_rate=best_trial_params['dropout_rate'],
                                    num_classes=self.num_classes).to(self.device)
        except TypeError as e:
            print("An error occurred while initializing the MLPClassifier. Please check that 'best_trial_params' is properly set.")
            # Optionally, re-raise the exception if you want the program to stop here
            raise e
        best_model.load_state_dict(best_model_state)

        return best_model, best_train_losses, best_val_losses, best_trial_params, best_model_test_loader, best_model_test_dataset
