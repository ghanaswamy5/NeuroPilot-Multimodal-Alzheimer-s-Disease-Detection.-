import torch

from preprocessing.mri_preprocess import preprocess_mri
from config.settings import LABELS

def predict_mri(model,file_path,device):

    x = preprocess_mri(file_path,device)

    with torch.no_grad():

        output = model(x)

        probs = torch.softmax(output,1)

    confidence,pred = torch.max(probs,1)

    prob_list = probs.squeeze().tolist()

    return {
        "prediction": LABELS[pred.item()],
        "confidence": round(confidence.item() * 100, 2),
        "probabilities": {
            "AD": round(prob_list[0] * 100, 2),
            "CN": round(prob_list[1] * 100, 2),
            "MCI": round(prob_list[2] * 100, 2),
        }
}