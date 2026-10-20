from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import DateTime

from datetime import datetime

from database.database import Base


class PredictionHistory(Base):

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)

    prediction_type = Column(String)

    prediction = Column(String)

    confidence = Column(Float)

    timestamp = Column(DateTime, default=datetime.utcnow)