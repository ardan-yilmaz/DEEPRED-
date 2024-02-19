import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader, random_split

class GOAnnotationsDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.prot_dict = {}
        self.go_terms = self.get_classes()
        self.num_classes = len(self.go_terms)
        self.fill_in_prot_dict()
        self.samples = [(protein_id, np.array(data['feature_vector']), np.array(data['binary_label_vector']))
                        for protein_id, data in self.prot_dict.items()]

    def fill_in_prot_dict(self):
        for go, go_index in self.go_terms.items():
            go_file = os.path.join(self.root_dir, go + '_filtered.h5')
            with h5py.File(go_file, 'r') as f:
                for protein_id, feat_vec in zip(f['ProteinIDs'], f['FeatureVectors']):
                    protein_id_str = protein_id.decode('utf-8')
                    if protein_id_str not in self.prot_dict:
                        self.prot_dict[protein_id_str] = {
                            'feature_vector': feat_vec[:],
                            'binary_label_vector': [0] * self.num_classes
                        }
                    self.prot_dict[protein_id_str]['binary_label_vector'][go_index] = 1

    def get_classes(self):
        classes = [f.replace('_filtered.h5', '') for f in os.listdir(self.root_dir) if f.endswith('_filtered.h5')]
        return {go: i for i, go in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        protein_id, feature_vector, binary_label_vector = self.samples[idx]
        feature_vector = torch.tensor(feature_vector, dtype=torch.float)
        binary_label_vector = torch.tensor(binary_label_vector, dtype=torch.float)
        return protein_id, feature_vector, binary_label_vector

class NormalizedDatasetWrapper(Dataset):
    def __init__(self, dataset, mean, std):
        self.dataset = dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        protein_id, feature_vector, binary_label_vector = self.dataset[idx]
        normalized_feature_vector = (feature_vector - self.mean) / (self.std + 1e-6)
        return protein_id, normalized_feature_vector, binary_label_vector



def calculate_normalization_params(samples):
    feature_vectors = torch.stack([sample[1].clone().detach() for sample in samples])
    mean = feature_vectors.mean(dim=0)
    std = feature_vectors.std(dim=0)
    return mean, std

def get_data_loaders(initial_dataset, val_portion=0.15, test_portion=0.20, batch_size=32, seed=42):
    torch.manual_seed(seed)

    # Split the dataset
    total_size = len(initial_dataset)
    test_size = int(test_portion * total_size)
    val_size = int(val_portion * total_size)
    train_size = total_size - val_size - test_size
    train_data, val_data, test_data = random_split(initial_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))

    # Calculate normalization parameters for each split
    train_mean, train_std = calculate_normalization_params(train_data)
    val_mean, val_std = calculate_normalization_params(val_data)
    test_mean, test_std = calculate_normalization_params(test_data)

    # Wrap each dataset split with the NormalizedDatasetWrapper using the calculated params
    train_dataset = NormalizedDatasetWrapper(train_data, train_mean, train_std)
    val_dataset = NormalizedDatasetWrapper(val_data, val_mean, val_std)
    test_dataset = NormalizedDatasetWrapper(test_data, test_mean, test_std)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



