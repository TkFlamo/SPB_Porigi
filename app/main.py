from fastapi import FastAPI, Request, Response
from model import model
from model import prepare
import pandas as pd

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.on_event("startup")
async def startup():
    await prepare()

@app.get("/address")
async def get_organization_page(query: str):
    data_pandas = pd.read_csv("additional_data/building_20230808.csv")
    result = model.model(query, prepare.full_vectors_dict, prepare.model, prepare.tokenizer)
    top_10 = [data_pandas[data_pandas["id"] == result[i]]["full_address"] for i in result]
    return {"target_building_id": result[0],
            "target_address": top_10[0],
            "top_10_address:" : top_10}
