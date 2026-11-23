from sqlalchemy import Column, Integer, String, Float
from app.db.session import Base

class Item(Base):
    __tablename__ = 'items'

    id = Column(Integer, primary_Key=True, index=True)
    name = Column(String, nullable=False)
    price = Column(Float, nullable=False)

