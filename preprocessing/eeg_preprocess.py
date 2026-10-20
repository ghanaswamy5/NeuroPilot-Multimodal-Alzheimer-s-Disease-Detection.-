import numpy as np
import torch

def preprocess_eeg(file_path, device):
    data = np.load(file_path)

    tensor = torch.tensor(
        data,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor.to(device)