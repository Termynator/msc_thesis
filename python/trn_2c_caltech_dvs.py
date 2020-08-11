import os
import sys

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

path = '/home/zeke/Programming/msc_thesis-master/'

import slayerSNN as snn
sys.path.append(path + 'python/slayerPytorch/src/')
from learningStats import learningStats

from data import caltechDataset,Network

data_path = 'dvs_data/Caltech101/'
model_path = 'models/'
yaml_path = 'yamls/'

netParams = snn.params(os.path.join(path,yaml_path,'2c_caltech_dvs.yaml'))

trainingSet = caltechDataset(datasetPath =netParams['training']['path']['in'], 
                             sampleFile  =netParams['training']['path']['train'],
                             samplingTime=netParams['simulation']['Ts'],
                             sampleLength=netParams['simulation']['tSample'])
trainLoader = DataLoader(dataset=trainingSet, batch_size=12, shuffle=False, num_workers=4)

testingSet = caltechDataset(datasetPath  =netParams['training']['path']['in'], 
                             sampleFile  =netParams['training']['path']['test'],
                             samplingTime=netParams['simulation']['Ts'],
                             sampleLength=netParams['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=12, shuffle=False, num_workers=4)

device = torch.device('cuda')
net = Network(netParams).to(device).float()
error = snn.loss(netParams).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)
stats = learningStats()

for epoch in range(200):
    tSt = datetime.now()
    
    # Training loop.
    for i, (input, target, label) in enumerate(trainLoader, 0):
        # Move the input and target to correct GPU.
        input  = input.to(device)
        target = target.to(device) 
        
        #print(input.shape)
        #print(target.shape)
        # Forward pass of the network.
        output = net.forward(input)

        # Gather the training stats.
        #print('label: ',label)
        #print('prediction: ',snn.predict.getClass(output))
        #print('target shape: ',target.shape)
        #print('output shape: ',output.shape)
        #print(torch.sum( snn.predict.getClass(output) == label ).data.item())# use prediciton to access corresping string label
        stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
        stats.training.numSamples     += len(label)
        
        # Calculate loss.
        print('output.shape: ',output.shape)
        print('target.shape: ',target.shape)
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
stats.save('stats_2c_slayer_dvs_cal')
torch.save(net,os.path.join(model_path,'2c_slayer_dvs_cal.net'))
net = torch.load(os.path.join(model_path,'2c_slayer_dvs_cal.net'))
