from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def home():
    return {'hello': 'world'}

@app.get('item/{item_id}')
def get_item(item_id: int):
    return {
        'item_id': item_id,
        'name': f'{item_id}',
    }

from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

@app.post("/items")
def create_item(item: Item):
    return {"received": item, "total_with_vat": item.price * 1.1}


