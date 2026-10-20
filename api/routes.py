from fastapi import APIRouter, UploadFile, File
import shutil
import os

from services.model_loader import load_models
from services.eeg_service import predict_eeg
from services.mri_service import predict_mri
from services.fusion_service import weighted_fusion
from config.settings import DEVICE

router = APIRouter()

# Load models only once when the server starts
eeg_model, mri_model = load_models()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/predict/eeg")
async def eeg_prediction(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_eeg(
        eeg_model,
        file_path,
        DEVICE
    )

    return result


@router.post("/predict/mri")
async def mri_prediction(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_mri(
        mri_model,
        file_path,
        DEVICE
    )

    return result

@router.post("/predict/fusion")
async def fusion_prediction(
    eeg_file: UploadFile = File(...),
    mri_file: UploadFile = File(...)
):

    eeg_path = os.path.join(UPLOAD_DIR, eeg_file.filename)
    mri_path = os.path.join(UPLOAD_DIR, mri_file.filename)

    with open(eeg_path, "wb") as buffer:
        shutil.copyfileobj(eeg_file.file, buffer)

    with open(mri_path, "wb") as buffer:
        shutil.copyfileobj(mri_file.file, buffer)

    eeg_result = predict_eeg(
        eeg_model,
        eeg_path,
        DEVICE
    )

    mri_result = predict_mri(
        mri_model,
        mri_path,
        DEVICE
    )

    fusion_result = weighted_fusion(
        eeg_result["probabilities"],
        mri_result["probabilities"]
    )

    return {
        "eeg_result": eeg_result,
        "mri_result": mri_result,
        "fusion_result": fusion_result
    }