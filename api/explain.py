from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os

from explainability.gradcam import generate_gradcam

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/explain/mri")
async def explain_mri(file: UploadFile = File(...)):
    image_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output = generate_gradcam(image_path)

    return FileResponse(
        output,
        media_type="image/png"
    )