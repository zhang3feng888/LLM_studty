# 请求体
# uvicorn Requestor:app --reload
# http://localhost:8000/docs

from typing import Union
from fastapi import FastAPI

# 步骤1：导入Pydantic 的BaseModel
from pydantic import BaseModel

# 步骤2：创建数据模型
class Item(BaseModel): # 继承自 BaseModel 的类
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None

app = FastAPI()
 
# 要发送数据，你必须使用下列方法之一：POST（较常见）、PUT、DELETE 或 PATCH
@app.post("/items/")
async def create_item(item: Item): # 步骤3：声明请求体为参数，添加到路径操作中
    return item

@app.post("/items_tax/")
async def create_item_tax(item: Item):
    item_dict = item.dict() # 另一个步骤3：使用模型——访问模型对象的所有属性
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict