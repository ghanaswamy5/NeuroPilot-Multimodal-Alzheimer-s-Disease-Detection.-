import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 📁 Path
DATA_PATH = "data/EEG/processed"
classes = ["AD", "CN", "MCI"]

# 🧠 Dataset
class EEGDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []

        for label, cls in enumerate(classes):
            folder = os.path.join(data_path, cls)
            files = os.listdir(folder)

            print(f"\nLoading {cls} data...")

            for i, file in enumerate(files):
                self.samples.append((os.path.join(folder, file), label))

                if i % 5000 == 0:
                    print(f"{cls}: loaded {i} files")

        print(f"\nTotal samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.load(file_path)

        return torch.tensor(data, dtype=torch.float32), label

# 🧠 Model
class EEGTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_proj = nn.Linear(16, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# 🚀 Main execution
if __name__ == "__main__":

    dataset = EEGDataset(DATA_PATH)

    # ✅ Train/Test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = EEGTransformer().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 🔥 Training
    epochs = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        print(f"\n🚀 Starting Epoch {epoch+1}")

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # ✅ Progress every 5000 batches
            if batch_idx % 5000 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        print(f"\nEpoch {epoch+1} Completed, Total Loss: {total_loss:.4f}")

        # ✅ Evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)

                outputs = model(data)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"✅ Test Accuracy: {accuracy:.2f}%")

    # 💾 Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/eeg_model.pth")

    print("\n✅ EEG Training Completed & Model Saved!")


    print("\n✅ EEG Training Completed & Model Saved!")

# 🔍 CHECK OUTPUT (Predictions vs Actual)
label_map = {0: "AD", 1: "CN", 2: "MCI"}

model.eval()
with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs, 1)

        for i in range(10):
            print(f"Pred: {label_map[predicted[i].item()]} | Actual: {label_map[labels[i].item()]}")

        break