import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class Simple_CNN(nn.Module):
    def __init__(self, dropOutVal=0.2):
        super(Simple_CNN, self).__init__()
        self.network = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128x3 -> 64x64x64

            # Second Convolutional Block
            nn.Dropout2d(dropOutVal),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64x128 -> 32x32x128

            # Third Convolutional Block
            nn.Dropout2d(dropOutVal),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32x128 -> 16x16x128

            # Fourth Convolutional Block
            nn.Dropout2d(dropOutVal),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16x64 -> 8x8x64

            # Flatten and Fully Connected Layers
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),  # Adjusted to the correct dimension after pooling
            nn.ReLU(),
            nn.Linear(256, 2),  # Output layer for binary classification
        )

    def forward(self, x):
        return self.network(x)



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_correct = 0
    total_samples = 0  # Track the total number of samples in this epoch
    mean_loss = 0
    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        mean_loss += loss
        counter += 1
        # Compute how many were correctly classified
        predicted = output.argmax(1)
        train_correct += (target == predicted).sum().cpu().item()
        total_samples += target.size(0)  # Add the batch size to the total
    
    # Return the accuracy for this epoch
    return (train_correct / total_samples),  mean_loss/counter


