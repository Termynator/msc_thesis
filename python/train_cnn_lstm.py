
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from scipy.ndimage import zoom
import torch.optim as optim

# --- 1. Custom PyTorch Dataset for 4D data ---
class NiiGz_4D_Dataset(Dataset):
    """Loads and preprocesses 4D .nii.gz fMRI files."""
    def __init__(self, filepaths, labels, spatial_target_shape=(50, 60, 50), temporal_downsample=2):
        self.filepaths = filepaths
        self.labels = labels
        self.spatial_target_shape = spatial_target_shape
        self.temporal_downsample = temporal_downsample

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        try:
            nii_img = nib.load(filepath)
            data = nii_img.get_fdata(dtype=np.float32)
        except Exception as e:
            print(f"\nError loading file: {filepath}\n{e}")
            # Return a dummy tensor to prevent crashing the whole batch
            return torch.zeros((10, 1, *self.spatial_target_shape)), torch.tensor(0)

        # --- Preprocessing ---
        # 1. Temporal Downsampling
        data = data[..., ::self.temporal_downsample]
        
        # 2. Spatial Resizing for each time point
        current_spatial_shape = data.shape[:-1]
        n_timepoints = data.shape[-1]
        zoom_factors = [t / s for t, s in zip(self.spatial_target_shape, current_spatial_shape)]
        
        resized_data = np.zeros(self.spatial_target_shape + (n_timepoints,), dtype=np.float32)
        for t in range(n_timepoints):
            resized_data[..., t] = zoom(data[..., t], zoom_factors, order=1)
        
        data = resized_data

        # 3. Normalization (to [0, 1] range)
        min_val, max_val = data.min(), data.max()
        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val)
        
        # 4. Permute and convert to tensor -> (T, C, D, H, W)
        data = np.transpose(data, (3, 0, 1, 2)) # T, H, W, D
        data_tensor = torch.from_numpy(data).float().unsqueeze(1) # T, 1, H, W, D
        data_tensor = data_tensor.permute(0, 1, 4, 2, 3) # T, 1, D, H, W

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return data_tensor, label_tensor

# --- 2. CNN-LSTM Model Definition ---
class CNN_LSTM(nn.Module):
    def __init__(self, cnn_output_size=32, lstm_hidden_size=64, num_classes=2):
        super(CNN_LSTM, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, cnn_output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )
        
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, D, H, W = x.size()
        
        c_in = x.view(batch_size * timesteps, C, D, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        
        lstm_out, _ = self.lstm(r_in)
        
        last_lstm_out = lstm_out[:, -1, :]
        output = self.fc(last_lstm_out)
        return output

# --- 3. Main Training Function ---
def main():
    start_time = time.time()
    print("Setting up...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = "/disks/Programming/msc_thesis-master/"
    numpys_dir = os.path.join(project_root, "numpys")

    X_train_paths = np.load(os.path.join(numpys_dir, 'mdd_X_train.npy'))
    y_train = np.load(os.path.join(numpys_dir, 'mdd_y_train.npy'))
    X_test_paths = np.load(os.path.join(numpys_dir, 'mdd_X_test.npy'))
    y_test = np.load(os.path.join(numpys_dir, 'mdd_y_test.npy'))

    train_dataset = NiiGz_4D_Dataset(X_train_paths, y_train)
    test_dataset = NiiGz_4D_Dataset(X_test_paths, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CNN_LSTM().to(device)

    num_controls = np.sum(y_train == 0)
    num_depr = np.sum(y_train == 1)
    class_weights = torch.tensor([len(y_train) / num_controls, len(y_train) / num_depr], dtype=torch.float).to(device)
    
    print(f"Class weights for loss function: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    print(f"\nStarting training for {num_epochs} epochs...")
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
    end_time = time.time()
    print(f"--- Total Execution Time: {(end_time - start_time) / 60:.2f} minutes ---")

if __name__ == '__main__':
    main()
