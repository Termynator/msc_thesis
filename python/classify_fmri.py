
"""
This script classifies the preprocessed fMRI data from the CAT-D dataset
using a traditional machine learning approach.

The script performs the following steps:
1.  Loads the preprocessed fMRI data and corresponding labels.
2.  Extracts features from the fMRI data using brain atlas parcellation.
3.  Trains a Linear Support Vector Machine (SVM) classifier.
4.  Evaluates the classifier's performance using cross-validation.
"""

import os
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# --- --- --- --- --- --- --- --- --- --- 
# Configuration
# --- --- --- --- --- --- --- --- --- --- 

# Directory where the preprocessed data is stored
PREPROCESSED_DATA_DIR = os.path.abspath('derivatives/preprocessing')
# Directory of the original dataset, to get the labels
DATA_DIR = os.path.abspath('datasets')

# --- --- --- --- --- --- --- --- --- --- 
# Data loading
# --- --- --- --- --- --- --- --- --- --- 

def load_data():
    """
    Loads the preprocessed fMRI data and the corresponding labels.
    """
    subjects = [s for s in os.listdir(PREPROCESSED_DATA_DIR) if s.startswith('subject_id_')]
    
    fmri_files = []
    labels = []

    for subject in subjects:
        subject_id = subject.split('_')[1]
        
        # Load the fMRI data
        fmri_file = os.path.join(PREPROCESSED_DATA_DIR, subject, 'smoothing', 'sub-{}_ses-v1_task-rest_bold_smooth.nii.gz'.format(subject_id))
        if os.path.exists(fmri_file):
            fmri_files.append(fmri_file)

            # Load the label
            label_file = os.path.join(DATA_DIR, 'sub-{}'.format(subject_id), 'sub-{}_sessions.tsv'.format(subject_id))
            if os.path.exists(label_file):
                session_data = pd.read_csv(label_file, sep='\t')
                # We assume the first session is the one we are interested in
                label = session_data.loc[session_data['session_id'] == 'ses-v1', 'c_ksadsdx_dx_detailed'].iloc[0]
                labels.append(label)
            else:
                # remove fmri file if no label
                fmri_files.pop()


    return fmri_files, labels

# --- --- --- --- --- --- --- --- --- --- 
# Feature extraction
# --- --- --- --- --- --- --- --- --- --- 

def extract_features(fmri_files):
    """
    Extracts features from the fMRI data using a brain atlas.
    """
    # Use the MSDL atlas
    atlas = datasets.fetch_atlas_msdl()
    atlas_filename = atlas.maps

    # Create a masker
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, memory='nilearn_cache', verbose=5)

    # Extract time series
    time_series = masker.fit_transform(fmri_files)
    
    # Calculate correlation matrices
    correlation_matrices = [np.corrcoef(ts.T) for ts in time_series]
    
    # Flatten the upper triangle of the correlation matrices to create a feature vector
    features = [mat[np.triu_indices_from(mat, k=1)] for mat in correlation_matrices]

    return np.array(features)

# --- --- --- --- --- --- --- --- --- --- 
# Classification
# --- --- --- --- --- --- --- --- --- --- 

if __name__ == '__main__':
    print("Loading data...")
    fmri_files, labels = load_data()
    
    print("Extracting features...")
    features = extract_features(fmri_files)
    
    # Convert labels to numerical format (0 for HV, 1 for others)
    numerical_labels = [0 if label == 'HV' else 1 for label in labels]
    
    print("Training and evaluating the classifier...")
    
    # Use a Linear SVM
    svm = SVC(kernel='linear')
    
    # Use stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=5)
    
    accuracies = []
    
    for train_index, test_index in cv.split(features, numerical_labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = np.array(numerical_labels)[train_index], np.array(numerical_labels)[test_index]
        
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        print(f"Fold accuracy: {accuracy}")

    print(f"\nAverage accuracy: {np.mean(accuracies):.2f} (+/- {np.std(accuracies):.2f})")
