import os
import h5py
import glob

# Define the root directory of the dataset
root_dir = 'dataset'

# Criteria for deletion based on directory names
criteria = {
    'MF': 50,
    'CC': 50,
    'BP': 100,
}

def check_and_delete_files(directory, min_protein_ids):
    # Search for .h5 files within the directory and its subdirectories
    for filepath in glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True):
        try:
            with h5py.File(filepath, 'r') as file:
                # Check if ProteinIDs exists and its length
                ll = len(file['ProteinIDs']) 
                if 'ProteinIDs' in file and ll < min_protein_ids:
                    print(f"To delete {filepath} with {ll} proteins: less than {min_protein_ids} ProteinIDs")
                    # os.remove(filepath)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

# Iterate over the criteria dictionary to apply the check and delete operation
for category, min_ids in criteria.items():
    category_dir = os.path.join(root_dir, category)
    check_and_delete_files(category_dir, min_ids)

print("Processing complete.")
