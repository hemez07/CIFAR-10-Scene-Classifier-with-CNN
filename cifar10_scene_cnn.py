import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load CIFAR-10 Dataset
train_set = CIFAR10(root="./data", train=True, download=True,
transform=transform)
test_set = CIFAR10(root="./data", train=False, download=True,
transform=transform)

# Split training into train and validation sets
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, val_set = random_split(train_set, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Show sample image
sample_image, sample_label = train_set[0]
plt.imshow(sample_image.permute(1, 2, 0))
plt.title(f"Class: {sample_label}")
plt.axis("off")
plt.show()



import torch.nn as nn
import torch.optim as optim

# Define a CNN architecture
class SceneCNN(nn.Module):
    def __init__(self, num_classes):
        super(SceneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 64 * 64, num_classes)  # 128x128 →64x64 after pooling

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x

# Initialize model
num_classes = 10  # CIFAR-10 has 10 classes
model = SceneCNN(num_classes)
print(model)




# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion,
epochs=5):
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}")

# Train the model
train_model(model, train_loader, val_loader, optimizer, criterion)



def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot results
plt.bar(["Test Accuracy"], [test_accuracy])
plt.ylabel("Accuracy")
plt.title("CNN Model Performance")
plt.show()