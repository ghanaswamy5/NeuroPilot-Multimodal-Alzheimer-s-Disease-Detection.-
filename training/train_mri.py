import torch
import torch.nn as nn
import torch.optim as optim

from models.mri_model import MRI_Model
from utils.data_loader import get_mri_loaders

# Paths
train_path = "data/MRI/train"
test_path = "data/MRI/test"

# Load data
train_loader, test_loader, classes = get_mri_loaders(train_path, test_path)

print("Classes:", classes)

# Model
model = MRI_Model()

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "checkpoints/mri_model.pth")

print("Model saved ✅")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print("Test Accuracy:", accuracy)