from fastapi import FastAPI
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from backend_utils import *
app = FastAPI()
# uvicorn backend_formal:app --reload
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
data_loc = "./data/"
algo = Algo(GWEA(data=EAData(loc=data_loc)))
@app.get("/")
async def root():
    return json.dumps({"message": "Hello Hello World"})

@app.get("/get_content/")
async def read_item(request: Request):
    query_params = request.query_params
    print(query_params['round'])
    return {"content":1}

@app.get("/get_table_data/")
async def read_table_data(request: Request):
    query_params = request.query_params
    res = algo.get_table_data(int(query_params['round']))
    return res

@app.get("/get_sim_data/")
async def read_sim_data(request: Request):
    query_params = request.query_params
    res = algo.get_sim_data(int(query_params['ID1']))
    return res

@app.get("/get_force_graph_data/")
async def read_graph_data(request: Request):
    query_params = request.query_params
    res = algo.get_force_graph_data(int(query_params['ID1']),int(query_params['ID2']))
    return res