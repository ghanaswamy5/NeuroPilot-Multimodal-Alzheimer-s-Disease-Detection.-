# import torch
# import numpy as np
# import cv2
# import os
# import sys

# # Fix import path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from models.eeg_model import EEGTransformer
# from models.mri_model import MRI_Model

# # -------------------------------
# # DEVICE
# # -------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print("Using device:", device)

# # -------------------------------
# # LOAD MODELS (NO WARNING ✅)
# # -------------------------------
# eeg_model = EEGTransformer().to(device)
# eeg_model.load_state_dict(
#     torch.load("checkpoints/eeg_model.pth", map_location=device, weights_only=True)
# )
# eeg_model.eval()

# mri_model = MRI_Model().to(device)
# mri_model.load_state_dict(
#     torch.load("checkpoints/mri_model.pth", map_location=device, weights_only=True)
# )
# mri_model.eval()

# # -------------------------------
# # LABELS
# # -------------------------------
# label_map = {0: "AD", 1: "CN", 2: "MCI"}

# # -------------------------------
# # LOAD EEG DATA
# # -------------------------------
# eeg_path = "data/EEG/processed/AD/sample_336639.npy"  

# eeg_data = np.load(eeg_path)
# eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).to(device)

# # -------------------------------
# # LOAD MRI IMAGE
# # -------------------------------
# mri_path = "data/MRI/test/AD/OAS1_0308_MR1_mpr-1_159.jpg"  # 🔁 change path

# img = cv2.imread(mri_path, 0)

# if img is None:
#     raise ValueError(" MRI image not found. Check path!")

# img = cv2.resize(img, (224, 224))
# img = img / 255.0

# img = np.expand_dims(img, axis=0)  # channel
# img = np.expand_dims(img, axis=0)  # batch

# mri_tensor = torch.tensor(img, dtype=torch.float32).to(device)

# # -------------------------------
# # GET OUTPUTS
# # -------------------------------
# with torch.no_grad():
#     eeg_output = eeg_model(eeg_tensor)
#     mri_output = mri_model(mri_tensor)

# # -------------------------------
# # SOFTMAX
# # -------------------------------
# eeg_probs = torch.softmax(eeg_output, dim=1)
# mri_probs = torch.softmax(mri_output, dim=1)

# # -------------------------------
# # PRINT INDIVIDUAL RESULTS
# # -------------------------------
# print("\n==============================")
# print("EEG RESULT")
# print("==============================")
# for i, label in label_map.items():
#     print(f"{label} : {eeg_probs[0][i].item()*100:.2f}%")

# print("\n==============================")
# print("MRI RESULT")
# print("==============================")
# for i, label in label_map.items():
#     print(f"{label} : {mri_probs[0][i].item()*100:.2f}%")

# # -------------------------------
# #  WEIGHTED FUSION (NO MODEL NEEDED)
# # -------------------------------
# w_eeg = 0.4
# w_mri = 0.6

# final_probs = (w_eeg * eeg_probs) + (w_mri * mri_probs)

# confidence, pred = torch.max(final_probs, 1)

# # -------------------------------
# # FINAL RESULT
# # -------------------------------
# print("\n==============================")
# print("FINAL FUSION RESULT")
# print("==============================")

# for i, label in label_map.items():
#     print(f"{label} : {final_probs[0][i].item()*100:.2f}%")

# print("\nFinal Prediction:", label_map[pred.item()])
# print(f"Confidence: {confidence.item()*100:.2f}%")



from services.model_loader import load_models
from services.eeg_service import predict_eeg
from services.mri_service import predict_mri
from services.fusion_service import weighted_fusion

from config.settings import DEVICE

eeg_model, mri_model = load_models()

eeg = predict_eeg(
    eeg_model,
    "sample.npy",
    DEVICE
)

mri = predict_mri(
    mri_model,
    "sample.jpg",
    DEVICE
)

fusion = weighted_fusion(
    eeg["probabilities"],
    mri["probabilities"]
)

print(eeg)
print(mri)
print(fusion)