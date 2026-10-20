import torch
import torch.nn as nn
import torch.optim as optim

from models.fusion_model import FusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FusionModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔥 Dummy training (replace later with real outputs)
for epoch in range(5):
    eeg = torch.rand(32, 3).to(device)   # EEG probs
    mri = torch.rand(32, 3).to(device)   # MRI probs
    labels = torch.randint(0, 3, (32,)).to(device)

    optimizer.zero_grad()

    output = model(eeg, mri)
    loss = criterion(output, labels)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "checkpoints/fusion_model.pth")
print("✅ Fusion model trained")