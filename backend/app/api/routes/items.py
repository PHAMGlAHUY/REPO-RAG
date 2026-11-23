from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.deps import get_db
from app.schemas.item import ItemCreate, ItemOut
from app.services import item_service

router = APIRouter(prefix="/items", tags=["items"])

@router.post("/", response_model=ItemOut)
def create(data: ItemCreate, db: Session = Depends(get_db)):
    return item_service.create_item(db, data)

@router.get("/", response_model=list[ItemOut])
def read_all(db: Session = Depends(get_db)):
    return item_service.list_items(db)