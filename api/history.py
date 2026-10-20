from fastapi import APIRouter
from fastapi import Depends

from sqlalchemy.orm import Session

from database.database import get_db
from database.models import PredictionHistory

router = APIRouter()


# @router.post("/history")

# def save_prediction(data: dict, db: Session = Depends(get_db)):

#     row = PredictionHistory(

#         prediction_type=data["prediction_type"],

#         prediction=data["prediction"],

#         confidence=data["confidence"]

#     )

#     db.add(row)

#     db.commit()

#     db.refresh(row)

#     return {"message": "Saved successfully"}

@router.post("/history")
def save_prediction(data: dict, db: Session = Depends(get_db)):
    print("Received:", data)

    row = PredictionHistory(
        prediction_type=data["prediction_type"],
        prediction=data["prediction"],
        confidence=data["confidence"]
    )

    db.add(row)
    db.commit()
    db.refresh(row)

    print("Saved ID:", row.id)

    return {
        "message": "Saved successfully",
        "id": row.id
    }


@router.get("/history")

def get_history(db: Session = Depends(get_db)):

    rows = db.query(PredictionHistory).order_by(
        PredictionHistory.id.desc()
    ).all()

    return rows


@router.delete("/history/{id}")

def delete_history(id: int, db: Session = Depends(get_db)):

    row = db.query(PredictionHistory).filter(
        PredictionHistory.id == id
    ).first()

    if row:

        db.delete(row)

        db.commit()

    return {"message": "Deleted"}