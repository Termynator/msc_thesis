

import os
import time
import numpy as np
from nilearn.decomposition import CanICA
from nilearn.connectome import ConnectivityMeasure
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from nilearn import image
from nilearn.datasets import fetch_icbm152_2009

def run_ica_svm():
    start_time = time.time()
    print("Starting ICA + SVM classification pipeline...")

    # --- 1. Load Data Splits ---
    project_root = "/disks/Programming/msc_thesis-master/"
    numpys_dir = os.path.join(project_root, "numpys")
    
    print("Loading data splits...")
    X_train_paths = np.load(os.path.join(numpys_dir, 'mdd_X_train.npy'))
    y_train = np.load(os.path.join(numpys_dir, 'mdd_y_train.npy'))
    X_test_paths = np.load(os.path.join(numpys_dir, 'mdd_X_test.npy'))
    y_test = np.load(os.path.join(numpys_dir, 'mdd_y_test.npy'))

    # --- 2. Resample data to a common template ---
    print("Fetching standard brain template...")
    template = fetch_icbm152_2009()

    def resample_files(filepaths, template_img):
        print(f"Resampling {len(filepaths)} files to MNI template space...")
        resampled_files = []
        resampled_dir = os.path.join(project_root, "resampled_fmri")
        os.makedirs(resampled_dir, exist_ok=True)

        for i, filepath in enumerate(filepaths):
            print(f"  Resampling subject {i+1}/{len(filepaths)}...", end='\r')
            output_filename = os.path.join(resampled_dir, os.path.basename(filepath))
            if not os.path.exists(output_filename):
                resampled_img = image.resample_to_img(filepath, template_img)
                resampled_img.to_filename(output_filename)
            resampled_files.append(output_filename)
        print("\nResampling complete.")
        return resampled_files

    X_train_paths_resampled = resample_files(X_train_paths, template['t1'])
    X_test_paths_resampled = resample_files(X_test_paths, template['t1'])

    # --- 3. Fit Group ICA Model ---
    print("Fitting Group ICA model on resampled training data... (This may take a while)")
    canica = CanICA(n_components=20,
                    memory="nilearn_cache",
                    memory_level=2,
                    n_jobs=1, # Using single core for memory stability
                    random_state=42)
    canica.fit(X_train_paths_resampled[:20]) # Using a subset to fit for memory reasons
    print("ICA model fitted.")

    # --- 4. Extract Features (FNC Matrices) ---
    def extract_features(filepaths, ica_model):
        print(f"Extracting features for {len(filepaths)} subjects...")
        connectivity_measure = ConnectivityMeasure(kind='correlation', vectorize=True)
        all_features = []
        for i, filepath in enumerate(filepaths):
            print(f"  Processing subject {i+1}/{len(filepaths)}: {os.path.basename(filepath)}", end='\r')
            time_series = ica_model.transform([filepath])
            feature_vector = connectivity_measure.fit_transform(time_series)[0]
            all_features.append(feature_vector)
        print("\nFeature extraction complete.")
        return np.array(all_features)

    X_train_features = extract_features(X_train_paths_resampled, canica)
    X_test_features = extract_features(X_test_paths_resampled, canica)

    # --- 5. Train SVM Classifier ---
    print("\nTraining SVM classifier...")
    svm = SVC(kernel='linear', class_weight='balanced', random_state=42)
    svm.fit(X_train_features, y_train)
    print("SVM training complete.")

    # --- 6. Evaluate the Model ---
    print("\nEvaluating model on the test set...")
    y_pred = svm.predict(X_test_features)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['control', 'depr'])

    print(f"\nTest Set Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    end_time = time.time()
    print(f"--- Total Execution Time: {(end_time - start_time) / 60:.2f} minutes ---")

if __name__ == '__main__':
    run_ica_svm()
