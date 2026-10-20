# import torch

# from config.settings import *

# def weighted_fusion(eeg_probs, mri_probs):

#     eeg_probs = torch.tensor(eeg_probs)
#     mri_probs = torch.tensor(mri_probs)

#     final = EEG_WEIGHT * eeg_probs + MRI_WEIGHT * mri_probs

#     confidence, pred = torch.max(final,0)

#     return {

#         "prediction": LABELS[pred.item()],
#         "confidence": confidence.item()*100,
#         "probabilities": final.tolist()

#     }

import torch

LABELS = ["AD", "CN", "MCI"]

def weighted_fusion(eeg_probs, mri_probs, eeg_weight=0.4, mri_weight=0.6):

    eeg_tensor = torch.tensor(
        [eeg_probs[label] for label in LABELS],
        dtype=torch.float32
    )

    mri_tensor = torch.tensor(
        [mri_probs[label] for label in LABELS],
        dtype=torch.float32
    )

    fused = eeg_weight * eeg_tensor + mri_weight * mri_tensor

    pred = torch.argmax(fused).item()

    return {
        "prediction": LABELS[pred],
        "confidence": float(fused[pred]),
        "probabilities": {
            label: round(float(fused[i]), 2)
            for i, label in enumerate(LABELS)
        }
    }