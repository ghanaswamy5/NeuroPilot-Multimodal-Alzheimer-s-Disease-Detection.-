import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from models.eeg_model import EEGTransformer   # ✅ IMPORTANT

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# ✅ Load model (same architecture as training)
model = EEGTransformer().to(device)
model.load_state_dict(
    torch.load("checkpoints/eeg_model.pth", map_location=device, weights_only=True)
)
model.eval()

# ✅ Label map
label_map = {0: "AD", 1: "CN", 2: "MCI"}

# ✅ Load real EEG data
data = np.load(r"data/EEG/processed/AD/sample_0.npy")
# print("Data shape:", data.shape)

# ✅ Convert to tensor
input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

# ✅ Prediction
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, pred = torch.max(probs, 1)

# Extract probabilities
ad_prob = probs[0][0].item()
cn_prob = probs[0][1].item()
mci_prob = probs[0][2].item()

# 🧠 Alzheimer score (same logic as MRI)
ad_score = (ad_prob * 1.0) + (mci_prob * 0.5) + (cn_prob * 0.0)

# 📊 Output format
print("\n--- Prediction Results ---")
print(f"AD: {ad_prob * 100:.2f}%")
print(f"CN: {cn_prob * 100:.2f}%")
print(f"MCI: {mci_prob * 100:.2f}%")

print("\n--- Alzheimer’s Score ---")
print(f"Alzheimer’s Confidence: {ad_score * 100:.2f}%")
