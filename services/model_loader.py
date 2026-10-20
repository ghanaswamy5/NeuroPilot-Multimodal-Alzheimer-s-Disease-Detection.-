import torch

from models.eeg_model import EEGTransformer
from models.mri_model import MRI_Model

from config.settings import *

def load_models():

    eeg_model = EEGTransformer().to(DEVICE)

    eeg_model.load_state_dict(
        torch.load(
            EEG_MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    eeg_model.eval()

    mri_model = MRI_Model().to(DEVICE)

    mri_model.load_state_dict(
        torch.load(
            MRI_MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    mri_model.eval()

    return eeg_model, mri_model