# Protein Function Prediction Project

## Project Overview
This project aims to predict protein functions from sequence data using a multi-label classification approach. It employs deep learning models, specifically Multi-Layer Perceptrons (MLP), implemented in PyTorch. The project includes modules for data loading and preprocessing, model architecture and hyperparameter optimization, model evaluation, optimal threshold finding, and utility functions for various tasks.

## Data Loading and Preprocessing

### GOAnnotationsDataset
A PyTorch `Dataset` class for loading protein features and annotations, mapping protein IDs to feature vectors and binary label vectors.

### NormalizedDatasetWrapper
Applies normalization to feature vectors, ensuring standardized input for model training and evaluation.

### Utility Functions
Includes functions for calculating normalization parameters and splitting the dataset into train, validation, and test sets.

## Model Architecture and Hyperparameter Optimization

### MLPClassifier
Defines a multi-layer perceptron model for multi-label classification, dynamically constructed based on specified parameters.

### HyperparameterSearch
Automates the search for optimal hyperparameters, identifying the best model configuration by iterating through randomly selected parameters.

## Model Evaluation

### ModelEvaluator
Assesses the trained model's performance using specified thresholds, computing metrics such as accuracy, precision, recall, F1-score, and MCC.

## Optimal Threshold Finding

### ThresholdFinder
Determines optimal decision thresholds for multi-label classification, maximizing the Matthews Correlation Coefficient (MCC) for each class.

## Model Saving and Logging

### ModelSaver
Facilitates the saving of model components, including the model itself, thresholds, losses, metadata, and test datasets, alongside configuring basic logging to monitor the process.

## Utility Functions and Custom Encoder

### NumpyEncoder
A custom JSON encoder for converting NumPy data types into JSON-serializable formats.

### Utility Functions
Provides essential functions for model loading, threshold retrieval, class weight calculation, custom MCC calculation, and visualization of training/validation losses.

## Getting Started

To get started with this project, clone the repository, install the required dependencies, and follow the usage instructions for each component as outlined above.

## Contributions

Contributions to this project are welcome. Please refer to the contribution guidelines for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Credits to datasets, libraries, or tools used in the development of this project.

