from fastapi import FastAPI, HTTPException
import pandas as pd
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class Item(BaseModel):
    index: int

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

QUEUEPATH = f"{Path(__file__).parent}/data/queue.json"
@app.get('/getnews')
async def getqueue():
	df = pd.read_json(QUEUEPATH)
	return df.to_dict()

@app.post('/update')
async def update(item: Item):
	df = pd.read_json(QUEUEPATH)
	df = df.drop(item.index)
	df = df.reset_index(drop=True)

	df.to_json(QUEUEPATH)
