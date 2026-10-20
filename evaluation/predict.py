import torch
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mri_model import MRI_Model

# Load model
model = MRI_Model()
model.load_state_dict(torch.load("checkpoints/mri_model.pth"))
model.eval()

# Class names (VERY IMPORTANT: match training order)
classes = ['AD', 'CN', 'MCI']

# Load imagedata/MRI/test/CN/OAS1_0080_MR2_mpr-1_156.jpg
img_path = r"data/MRI/test/MCI/OAS1_0003_MR1_mpr-2_136.jpg"  
img = cv2.imread(img_path, 0)

# Preprocess (same as training)
img = cv2.resize(img, (224, 224))
img = img / 255.0

# Convert to tensor
img = np.expand_dims(img, axis=0)  # channel
img = np.expand_dims(img, axis=0)  # batch

img = torch.tensor(img, dtype=torch.float32)

# Prediction
with torch.no_grad():
    output = model(img)
    probs = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, 1)

print("\n--- Prediction Results ---")

for i, class_name in enumerate(classes):
    print(f"{class_name}: {probs[0][i].item() * 100:.2f}%")

# Extract probabilities
ad_prob = probs[0][classes.index('AD')].item()
mci_prob = probs[0][classes.index('MCI')].item()
cn_prob = probs[0][classes.index('CN')].item()

# Convert to Alzheimer’s score
ad_score = (ad_prob * 1.0) + (mci_prob * 0.5) + (cn_prob * 0.0)

print("\n--- Alzheimer’s Score ---")
print(f"Alzheimer’s Confidence: {ad_score * 100:.2f}%")