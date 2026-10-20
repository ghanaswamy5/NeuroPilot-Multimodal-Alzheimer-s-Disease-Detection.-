import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EEG_MODEL_PATH = "checkpoints/eeg_model.pth"
MRI_MODEL_PATH = "checkpoints/mri_model.pth"
FUSION_MODEL_PATH = "checkpoints/fusion_model.pth"

EEG_WEIGHT = 0.4
MRI_WEIGHT = 0.6

LABELS = {
    0: "AD",
    1: "CN",
    2: "MCI"
}