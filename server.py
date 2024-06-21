from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
import urllib.request
from get_output import get_output
import cloudinary
app = FastAPI()

cloudinary.config(cloud_name='<cloud-name-here>',
                  api_key='561963979883516',
                  api_secret='9QgwkPAa2apZpgjIRc4rfR-pTNw')

from torch import  optim 
from models.generator import Generator
import configs.config as config
from configs.utils import *
models = ["map","anime","cityscapes","edge2shoes","facades"]
generators = { i : Generator(in_channels=3).to(config.DEVICE) for i in models }
for i , model in generators.items():
    opt_gen  = optim.Adam(model.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.99)) 
    load_checkpoint(f"data/{i}/gen.pth.tar",model,opt_gen,config.LEARNING_RATE)
    generators[i] = model 

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/maps")
def read_item():
    return {"location":"here"}


# Define the Pydantic model for the request body
class ImageProcessRequest(BaseModel):
    model: str
    path: str

@app.post("/process_image")
def process_image(request: ImageProcessRequest):
    image_path = request.path
    model = request.model
    print(model)
    urllib.request.urlretrieve(image_path,"images/input.png")
    outPath = get_output(model=generators[model]) 
    print(outPath)
    return {"processed_image_path": outPath}
