import torch
import os

from search_hyperparams import HyperparameterSearch
from data_loader import get_data_loaders, GOAnnotationsDataset
from utils import get_class_weights, get_num_classes
from thresholds import ThresholdFinder
from model_eval import ModelEvaluator
from saver import ModelSaver

# get all the root_dirs
sub_graphs = ["dataset/MF, dataset/BP, dataset/CC"] 
MF_levels = ["dataset/MF/level" + str(i) for i in range(0, 10)]
BP_levels = ["dataset/BP/level" + str(i) for i in range(0, 13)]
CC_levels = ["dataset/CC/level" + str(i) for i in range(0, 11)]
all_root_dirs = MF_levels + BP_levels + CC_levels
all_root_dirs



# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for root_dir in all_root_dirs:
    try:
        # Splitting the root directory and constructing model path
        root_dir_l = root_dir.split("/")
        model_path = root_dir_l[1] + "_" + root_dir_l[2]
        save_path = os.path.join(root_dir, "saved_info")
        save_path = os.path.join("models", model_path)
        print(f"Model to be saved @ {save_path}")
    except Exception as e:
        print(f"Error preparing model save path: {e}")
        continue

    try:
        # Get the details of the dataset
        num_classes = get_num_classes(root_dir)
        initial_dataset = GOAnnotationsDataset(root_dir)
        class_weights = get_class_weights(initial_dataset)
    except Exception as e:
        print(f"Error loading dataset details: {e}")
        continue

    try:
        # Search for hyperparameters
        search = HyperparameterSearch(initial_dataset, num_classes, class_weights, num_trials=10, max_epochs=100, device=device)
        best_model, best_train_losses, best_val_losses, best_trial_params, best_model_test_loader, best_model_test_dataset = search.run_search()
    except Exception as e:
        print(f"Error during hyperparameter search: {e}")
        continue

    try:
        # Find optimal thresholds
        threshold_finder = ThresholdFinder(best_model, num_classes, device)
        optimal_thresholds = threshold_finder.find_optimal_threshold(best_model_test_loader)
    except Exception as e:
        print(f"Error finding optimal thresholds: {e}")
        continue

    try:
        # Evaluate the model
        evaluator = ModelEvaluator(best_model, optimal_thresholds, device=device)
        eval_metrics = evaluator.evaluate(best_model_test_loader)
    except Exception as e:
        print(f"Error evaluating the model: {e}")
        continue

    try:
        # Save the model
        saver = ModelSaver(save_path)
        saver.save(best_model, optimal_thresholds, best_trial_params, eval_metrics, best_train_losses, best_val_losses, best_model_test_dataset)
    except Exception as e:
        print(f"Error saving the model: {e}")
        continue
