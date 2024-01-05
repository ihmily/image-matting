# -*- coding: utf-8 -*-

import sys
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi import Request
import requests
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import numpy as np
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

app = FastAPI()

model_paths = {
    "universal": {'path': 'damo/cv_unet_universal-matting', 'task': Tasks.universal_matting},
    "people": {'path': 'damo/cv_unet_image-matting', 'task': Tasks.portrait_matting},
}

default_model = list(model_paths.keys())[0]
default_model_info = model_paths[default_model]
loaded_models = {default_model: pipeline(default_model_info['task'], model=default_model_info['path'])}

UPLOAD_FOLDER = "./upload"
OUTPUT_FOLDER = "./output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


class ModelLoader:
    def __init__(self):
        self.loaded_models = {default_model: loaded_models[default_model]}

    def load_model(self, model_name):
        if model_name not in self.loaded_models:
            model_info = model_paths[model_name]
            model_path = model_info['path']
            task_group = model_info['task']

            self.loaded_models[model_name] = pipeline(task_group, model=model_path)
        return self.loaded_models[model_name]


model_loader = ModelLoader()


@app.post("/switch_model/{new_model}")
async def switch_model(new_model: str):
    if new_model not in model_paths:
        return {"content": "Invalid model selection"}, 400
    model_info = model_paths[new_model]

    loaded_models[new_model] = pipeline(model_info['task'], model=model_info['path'])
    model_loader.loaded_models = loaded_models
    return {"content": f"Switched to model: {new_model}"}, 200


@app.post("/matting")
async def matting(image: UploadFile = File(...), model: str = Form(default=default_model, alias="model")):
    image_bytes = await image.read()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if model not in model_paths:
        return {"content": "Invalid model selection"}, 400

    selected_model = model_loader.load_model(model)

    filename = uuid.uuid4()
    original_image_filename = f"original_{filename}.png"
    image_filename = f"image_{filename}.png"
    mask_filename = f"mask_{filename}.png"

    cv2.imwrite(os.path.join(UPLOAD_FOLDER, original_image_filename), img)

    result = selected_model(img)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, image_filename), result[OutputKeys.OUTPUT_IMG])
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, mask_filename), result['output_img'][:, :, 3])

    response_data = {
        "result_image_url": f"/output/{image_filename}",
        "mask_image_url": f"/output/{mask_filename}",
        "original_image_size": {"width": img.shape[1], "height": img.shape[0]},
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return response_data


@app.post("/matting/url")
async def matting_url(request: Request, model: str = Form(default=default_model, alias="model")):
    try:
        json_data = await request.json()
        image_url = json_data.get("image_url")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing JSON data: {str(e)}")

    if not image_url:
        raise HTTPException(status_code=400, detail="Image URL is required")

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")

    if model not in model_paths:
        raise HTTPException(status_code=400, detail="Invalid model selection")

    selected_model = model_loader.load_model(model)

    filename = uuid.uuid4()
    original_image_filename = f"original_{filename}.png"
    image_filename = f"image_{filename}.png"
    mask_filename = f"mask_{filename}.png"

    cv2.imwrite(os.path.join(UPLOAD_FOLDER, original_image_filename), img)

    result = selected_model(img)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, image_filename), result[OutputKeys.OUTPUT_IMG])
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, mask_filename), result['output_img'][:, :, 3])

    response_data = {
        "result_image_url": f"/output/{image_filename}",
        "mask_image_url": f"/output/{mask_filename}",
        "original_image_size": {"width": img.shape[1], "height": img.shape[0]},
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return response_data


templates = Jinja2Templates(directory="web")
app.mount("/static", StaticFiles(directory="./web/static"), name="static")
app.mount("/output", StaticFiles(directory="./output"), name="output")
app.mount("/upload", StaticFiles(directory="./upload"), name="upload")


@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "default_model": default_model,
                                                     "available_models": list(model_paths.keys())})


if __name__ == "__main__":
    import uvicorn

    defult_bind_host = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"
    uvicorn.run(app, host=defult_bind_host, port=8000)
