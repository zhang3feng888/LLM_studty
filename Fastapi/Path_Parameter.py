# 路径参数
# uvicorn Path_Parameter:app --reload
# http://localhost:8000/docs

from fastapi import FastAPI
from enum import Enum

app = FastAPI()

"""
# http://127.0.0.1:8000
@app.get("/")
async def root():
    return {"message": "Hello World"}
"""

"""
# http://127.0.0.1:8000/items/afgaef
@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}
"""

"""
# http://127.0.0.1:8000/items/1111
# item_id 只能是整数，其他类型会报错
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
"""

""""""
# http://127.0.0.1:8000/models/lenet
class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"} # resnet

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
