import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
import numpy as np
import os
import time
from data import read_dataset

def conv2d_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        super(UNet2D, self).__init__()

        features = init_features
        self.encoder1 = UNet2D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet2D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet2D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet2D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(out_channels * 48 * 48, out_channels)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        output = self.conv(dec1)
        output = self.flatten(output)
        output = self.linear(output)
        return output

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

class caltechDvsDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath 
        self.samplingTime = samplingTime
        self.samples = np.load(sampleFile,allow_pickle = True)
        self.nTimeBins    = int(sampleLength / samplingTime)
        self.classes = np.sort(np.unique(self.samples[1,:]))


    def __getitem__(self, index):
        input_index  = self.samples[0,index]
        class_label  = self.samples[1,index]
        class_code = list(self.classes).index(class_label)

        events = read_dataset(input_index)
        frame = events.to_frame(self.nTimeBins)
        frame = torch.from_numpy(frame).float()
        frame = frame.unsqueeze(0)
        
        desiredClass = torch.tensor(class_code, dtype=torch.long)
        return frame, desiredClass, class_code

    def __len__(self):
        return len(self.samples[0,:])

if __name__ == '__main__':
    net_params = snn.params('yamls/unet_caltech_dvs.yaml')
    
    training_set = caltechDvsDataset(datasetPath  = net_params['training']['path']['in'], 
                               sampleFile   = net_params['training']['path']['train'],
                               samplingTime = net_params['simulation']['Ts'],
                               sampleLength = net_params['simulation']['tSample'])
    train_loader = DataLoader(dataset=training_set, batch_size=1, shuffle=False, num_workers=4)

    testing_set = caltechDvsDataset(datasetPath  = net_params['training']['path']['in'],
                                 sampleFile   = net_params['training']['path']['test'],
                                 samplingTime = net_params['simulation']['Ts'],
                                 sampleLength = net_params['simulation']['tSample'])
    test_loader = DataLoader(dataset=testing_set, batch_size=1, shuffle=False, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2D(in_channels=1, out_channels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    checkpoint_path = 'unet_checkpoint.pth'
    start_epoch = 0
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        patience_counter = checkpoint['patience_counter']
        print(f"Resuming training from epoch {start_epoch} with best loss {best_loss:.4f}")

    start_time = time.time()

    for epoch in range(start_epoch, 100):
        model.train()
        for i, (inputs, labels, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/100], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Evaluate on test set for early stopping
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/100], Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'patience_counter': patience_counter,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs due to no improvement in test loss.')
                break
    
    end_time = time.time()
    training_time = end_time - start_time

    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_unet_model.pth'))
    model.eval()
    final_correct = 0
    final_total = 0
    final_test_loss = 0.0
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            final_test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            final_total += labels.size(0)
            final_correct += (predicted == labels).sum().item()
    
    final_test_loss /= len(test_loader)
    final_accuracy = 100 * final_correct / final_total

    print(f'Final Test Loss: {final_test_loss:.4f}')
    print(f'Final Test Accuracy: {final_accuracy:.2f}%')
    print(f'Training Time: {training_time:.2f} seconds')

    with open('unet_performance_metrics.txt', 'w') as f:
        f.write(f'Final Test Loss: {final_test_loss:.4f}\n')
        f.write(f'Final Test Accuracy: {final_accuracy:.2f}%\n')
        f.write(f'Training Time: {training_time:.2f} seconds\n')