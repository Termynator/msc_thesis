import os
import sys
import glob

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

import slayerSNN as snn
from slayerSNN.learningStats import learningStats

from data import simtbDataset, Network

# Configuration parameters
DATA_PATH = 'simtb_ds/01_EXPERIMENT/spk_niis/'
NPY_PATH = 'numpys/'
MODEL_PATH = 'models/'
YAML_PATH = 'yamls/'
CHECKPOINT_FILE = 'slayer_checkpoint.pth'

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_SIZE = 0.8
TEST_SIZE = 0.2

# Construct full paths
checkpoint_path = os.path.join(MODEL_PATH, CHECKPOINT_FILE)

files = os.listdir(DATA_PATH)
labls = np.empty_like(files)

for i in range(len(labls)):
    labls[i] = files[i][0]
    files[i] = os.path.join(DATA_PATH, files[i])

trn_files, tst_files, trn_labls, tst_labls = train_test_split(files, labls, train_size=TRAIN_SIZE, random_state=SEED)

trn = np.stack([trn_files, trn_labls])
tst = np.stack([tst_files, tst_labls])

np.save(os.path.join(NPY_PATH, 'trn_simtb_dc.npy'), trn)
np.save(os.path.join(NPY_PATH, 'tst_simtb_dc.npy'), tst)

netParams = snn.params(os.path.join(YAML_PATH, 'simtb_dc.yaml'))
batch_size = 1

trainingSet = simtbDataset(datasetPath  = netParams['training']['path']['in'], 
                           sampleFile   = netParams['training']['path']['train'],
                           samplingTime = netParams['simulation']['Ts'],
                           sampleLength = netParams['simulation']['tSample'])
trainLoader = DataLoader(dataset=trainingSet, batch_size=batch_size, shuffle=False, num_workers=4)

testingSet = simtbDataset(datasetPath  = netParams['training']['path']['in'], 
                          sampleFile   = netParams['training']['path']['test'],
                          samplingTime = netParams['simulation']['Ts'],
                          sampleLength = netParams['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device('cuda')
net = Network(netParams).to(device).float()
error = snn.loss(netParams).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)
stats = learningStats()

start_epoch = 0
best_loss = float('inf')
patience = 5
patience_counter = 0
total_training_time = 0

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    stats = checkpoint['stats']
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    patience_counter = checkpoint['patience_counter']
    total_training_time = checkpoint['total_training_time']
    print(f"Resuming training from epoch {start_epoch} with best loss {best_loss:.4f}")

for epoch in range(start_epoch, 200):
    tSt = datetime.now()
    if epoch == start_epoch:
        total_training_time = 0
    
    # Training loop.
    for i, (input, target, label) in enumerate(trainLoader, 0):
        # Move the input and target to correct GPU.
        input  = input.to(device)
        target = target.to(device) 
        
        
        # Forward pass of the network.
        output = net.forward(input)

        
        stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
        stats.training.numSamples     += len(label)
        
        # Calculate loss.
        
        loss = error.numSpikes(output, target)
        
        # Reset gradients to zero.
        optimizer.zero_grad()
        
        # Backward pass of the network.
        loss.backward()
        
        # Update weights.
        optimizer.step()

        # Gather training loss stats.
        stats.training.lossSum += loss.cpu().data.item()

        # Display training stats.
        stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

    # Testing loop.
    # Same steps as Training loops except loss backpropagation and weight update.
    for i, (input, target, label) in enumerate(testLoader, 0):
        input  = input.to(device)
        target = target.to(device) 
        
        output = net.forward(input)

        stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
        stats.testing.numSamples     += len(label)

        loss = error.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().data.item()
        stats.print(epoch, i)
    
    # Update stats.
    stats.update()

    # Early stopping check
    current_test_loss = stats.testing.lossLog[-1]
    if current_test_loss < best_loss:
        best_loss = current_test_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats': stats,
            'best_loss': best_loss,
            'patience_counter': patience_counter,
            'total_training_time': total_training_time + (datetime.now() - tSt).total_seconds(),
        }, checkpoint_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs due to no improvement in test loss.')
            break
    
    total_training_time += (datetime.now() - tSt).total_seconds()

# Load best model for final evaluation
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    stats = checkpoint['stats']
    total_training_time = checkpoint['total_training_time']

final_test_loss = stats.testing.lossLog[-1]
final_accuracy = stats.testing.accuracyLog[-1]

print(f'Final Test Loss: {final_test_loss:.4f}')
print(f'Final Test Accuracy: {final_accuracy:.2f}%')
print(f'Training Time: {total_training_time:.2f} seconds')

with open('slayer_performance_metrics.txt', 'w') as f:
    f.write(f'Final Test Loss: {final_test_loss:.4f}\n')
    f.write(f'Final Test Accuracy: {final_accuracy:.2f}%\n')
    f.write(f'Training Time: {total_training_time:.2f} seconds\n')


# Plot the results.
plt.figure(1)
plt.semilogy(stats.training.lossLog, label='Training')
plt.semilogy(stats.testing .lossLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(stats.training.accuracyLog, label='Training')
plt.plot(stats.testing .accuracyLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Save

