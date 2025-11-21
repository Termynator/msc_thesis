import torch
import matplotlib.pyplot as plt
import os
from slayerSNN.learningStats import learningStats

MODEL_PATH = 'models/'
CHECKPOINT_FILE = 'slayer_caltech_checkpoint.pth'
checkpoint_path = os.path.join(MODEL_PATH, CHECKPOINT_FILE)

if not os.path.exists(checkpoint_path):
    print(f"Checkpoint file not found at {checkpoint_path}")
    print("Please run the training script first to generate a checkpoint.")
else:
    print(f"Loading stats from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    stats = checkpoint['stats']

    # Plot loss
    plt.figure(1)
    plt.semilogy(stats.training.lossLog, label='Training')
    plt.semilogy(stats.testing.lossLog, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)
    loss_path = 'report/imgs/training_loss.png'
    plt.savefig(loss_path)
    print(f"Saved loss graph to {loss_path}")

    # Plot accuracy
    plt.figure(2)
    plt.plot(stats.training.accuracyLog, label='Training')
    plt.plot(stats.testing.accuracyLog, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)
    accuracy_path = 'report/imgs/training_accuracy.png'
    plt.savefig(accuracy_path)
    print(f"Saved accuracy graph to {accuracy_path}")
