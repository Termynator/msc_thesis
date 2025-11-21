
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_mdd_data():
    """
    Parses the ds002748 dataset's participants.tsv file, creates file paths and labels,
    splits the data into training and testing sets, and saves them to .npy files.
    """
    # Define paths
    project_root = "/disks/Programming/msc_thesis-master/"
    dataset_dir = os.path.join(project_root, "ds002748-1.0.5")
    participants_file = os.path.join(dataset_dir, "participants.tsv")
    output_dir = os.path.join(project_root, "numpys")

    print("Starting data preparation...")

    # --- 1. Parse participants.tsv ---
    filepaths = []
    labels = []
    
    # Check if participants file exists
    if not os.path.exists(participants_file):
        print(f"ERROR: Cannot find participants file at {participants_file}")
        return

    with open(participants_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)  # Skip header
        
        # Find column indices
        try:
            id_idx = header.index('participant_id')
            group_idx = header.index('group')
        except ValueError as e:
            print(f"ERROR: Missing required column in participants.tsv: {e}")
            return

        for row in reader:
            participant_id = row[id_idx]
            group = row[group_idx]
            
            # Construct the filepath for the functional MRI data
            fmri_file = os.path.join(dataset_dir, participant_id, 'func', f'{participant_id}_task-rest_bold.nii.gz')
            
            if os.path.exists(fmri_file):
                filepaths.append(fmri_file)
                # Convert labels to numeric format (0 for control, 1 for depr)
                labels.append(1 if group == 'depr' else 0)
            else:
                print(f"Warning: Could not find fMRI file for {participant_id}: {fmri_file}")

    if not filepaths:
        print("ERROR: No valid fMRI files found. Please check the dataset directory.")
        return

    print(f"Found {len(filepaths)} subjects with corresponding fMRI files.")
    print(f"Class distribution: {np.sum(labels)} 'depr' and {len(labels) - np.sum(labels)} 'control'.")

    # --- 2. Split the data ---
    # Use stratify to maintain the same class distribution in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        filepaths, 
        labels, 
        test_size=0.2,       # 20% for testing
        random_state=42,     # for reproducibility
        stratify=labels
    )

    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # --- 3. Save the splits ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, 'mdd_X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'mdd_X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'mdd_y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'mdd_y_test.npy'), y_test)

    print(f"Successfully saved data splits to {output_dir}")

if __name__ == '__main__':
    prepare_mdd_data()
