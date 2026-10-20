import cv2
import numpy as np
import torch

def preprocess_mri(file_path, device):

    img = cv2.imread(file_path, 0)

    if img is None:
        raise Exception("Invalid MRI image.")

    img = cv2.resize(img, (224,224))

    img = img / 255.0

    img = np.expand_dims(img,0)
    img = np.expand_dims(img,0)

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    )

    return tensor.to(device)