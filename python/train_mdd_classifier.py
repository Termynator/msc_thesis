
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy.ndimage import zoom
import torch.optim as optim

# --- 1. Custom PyTorch Dataset ---
class NiiGzDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and preprocessing .nii.gz fMRI files.
    """
    def __init__(self, filepaths, labels, target_shape=(96, 112, 96)):
        self.filepaths = filepaths
        self.labels = labels
        self.target_shape = target_shape

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Load the fMRI data
        filepath = self.filepaths[idx]
        try:
            nii_img = nib.load(filepath)
            data = nii_img.get_fdata()
        except Exception as e:
            print(f"Error loading file: {filepath}")
            print(e)
            # Return a dummy tensor and label if file is corrupt
            return torch.zeros((1,) + self.target_shape), torch.tensor(0)

        # If data is 4D (like fMRI), average across the time dimension
        if data.ndim == 4:
            data = data.mean(axis=-1)

        # Preprocessing
        # 1. Resizing (Zooming)
        zoom_factors = [t / s for t, s in zip(self.target_shape, data.shape)]
        data = zoom(data, zoom_factors, order=1) # Using order=1 for linear interpolation

        # 2. Normalization (to [0, 1] range)
        min_val, max_val = data.min(), data.max()
        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val)
        
        # 3. Add channel dimension and convert to tensor
        data_tensor = torch.from_numpy(data).float().unsqueeze(0) # Shape: [1, D, H, W]
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return data_tensor, label_tensor

# --- 2. 3D CNN Model Definition ---
class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # Halves dimensions
            
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), # Halves dimensions again
            
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # Halves dimensions again
        )
        
        # Calculate the size of the flattened features after conv layers
        # Input: 96x112x96 -> After 3x MaxPool(2): (96/8)x(112/8)x(96/8) = 12x14x12
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 14 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2) # 2 output classes (control, depr)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- 3. Training and Evaluation Loop ---
def main():
    # --- Setup ---
    print("Setting up...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = "/disks/Programming/msc_thesis-master/"
    numpys_dir = os.path.join(project_root, "numpys")

    # Load data splits
    X_train_paths = np.load(os.path.join(numpys_dir, 'mdd_X_train.npy'))
    y_train = np.load(os.path.join(numpys_dir, 'mdd_y_train.npy'))
    X_test_paths = np.load(os.path.join(numpys_dir, 'mdd_X_test.npy'))
    y_test = np.load(os.path.join(numpys_dir, 'mdd_y_test.npy'))

    # Create Datasets and DataLoaders
    train_dataset = NiiGzDataset(X_train_paths, y_train)
    test_dataset = NiiGzDataset(X_test_paths, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2)

    # --- Model, Loss, and Optimizer ---
    model = Simple3DCNN().to(device)

    # Handle class imbalance
    # Calculate weights: weight = 1 / (number of samples in class)
    num_controls = np.sum(y_train == 0)
    num_depr = np.sum(y_train == 1)
    class_weights = torch.tensor([len(y_train) / num_controls, len(y_train) / num_depr], dtype=torch.float).to(device)
    
    print(f"Class weights for loss function: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # --- Training Loop ---
    num_epochs = 15
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}", end='\r')

        epoch_loss = running_loss / len(train_loader)
        
        # --- Evaluation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"\nEpoch {epoch+1}/{num_epochs} | Training Loss: {epoch_loss:.4f} | Test Accuracy: {accuracy:.2f}%")

    print("\nFinished Training")

if __name__ == '__main__':
    # PyTorch multiprocessing for num_workers > 0
    # Needs to be in the main block
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
