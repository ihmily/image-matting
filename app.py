# -*- coding: utf-8 -*-

import sys
import os
import uuid
from datetime import datetime
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
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
            if not model_info:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid model selection")
            model_path = model_info['path']
            task_group = model_info['task']

            self.loaded_models[model_name] = pipeline(task_group, model=model_path)
        return self.loaded_models[model_name]


model_loader = ModelLoader()


def get_filename():
    filename = uuid.uuid4()
    original_image_filename = f"original_{filename}.png"
    image_filename = f"image_{filename}.png"
    mask_filename = f"mask_{filename}.png"
    return original_image_filename, image_filename, mask_filename


# remove excess transparent background and crop the image
def crop_image_by_alpha_channel(input_image: np.ndarray | str, output_path: str):
    img_array = cv2.imread(input_image, cv2.IMREAD_UNCHANGED) if isinstance(input_image, str) else input_image
    if img_array.shape[2] != 4:
        raise ValueError("Input image must have an alpha channel")

    alpha_channel = img_array[:, :, 3]
    bbox = cv2.boundingRect(alpha_channel)
    x, y, w, h = bbox
    cropped_img_array = img_array[y:y + h, x:x + w]
    cv2.imwrite(output_path, cropped_img_array)
    return output_path


def process_image(image_bytes: bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    final_img = convert_image_to_white_background(image=img)
    if final_img is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image")
    return final_img


def convert_image_to_white_background(image_path: str = None, image: np.ndarray | None = None):
    try:
        if image_path is not None:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        elif image is not None:
            img = image
        else:
            raise ValueError("Either image_path or image must be provided.")

        if img.shape[2] == 4:
            alpha_channel = img[:, :, 3]
            rgb_channels = img[:, :, :3]

            alpha_channel_3d = alpha_channel[:, :, np.newaxis] / 255.0
            alpha_channel_3d = np.repeat(alpha_channel_3d, 3, axis=2)

            white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

            foreground = cv2.multiply(rgb_channels, alpha_channel_3d, dtype=cv2.CV_8UC3)
            background = cv2.multiply(white_background_image, 1 - alpha_channel_3d, dtype=cv2.CV_8UC3)

            final_img = cv2.add(foreground, background)
        else:
            final_img = img
        return final_img
    except Exception as e:
        print(f'Error: {e}')
        return None


@app.post("/switch_model/{new_model}")
async def switch_model(new_model: str):
    if new_model not in model_paths:
        return {"content": "Invalid model selection"}, status.HTTP_400_BAD_REQUEST
    model_info = model_paths[new_model]

    loaded_models[new_model] = pipeline(model_info['task'], model=model_info['path'])
    model_loader.loaded_models = loaded_models
    return {"content": f"Switched to model: {new_model}"}, status.HTTP_200_OK


@app.post("/matting")
async def matting(image: UploadFile = File(...), model: str = Form(default=default_model, alias="model")):
    try:
        image_bytes = await image.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

        if model not in model_paths:
            return {"content": "Invalid model selection"}, status.HTTP_400_BAD_REQUEST

        selected_model = model_loader.load_model(model)
        original_image_filename, image_filename, mask_filename = get_filename()
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, original_image_filename), img)

        final_img = convert_image_to_white_background(image=img)
        if final_img is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image")
        result = selected_model(final_img)

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, image_filename), result[OutputKeys.OUTPUT_IMG])
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, mask_filename), result[OutputKeys.OUTPUT_IMG][:, :, 3])

        response_data = {
            "code": 0,
            "result_image_url": f"/output/{image_filename}",
            "mask_image_url": f"/output/{mask_filename}",
            "original_image_size": {"width": img.shape[1], "height": img.shape[0]},
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return response_data
    except HTTPException as e:
        return {"error": str(e)}, e.status_code
    except Exception as e:
        return {"error": str(e)}, status.HTTP_500_INTERNAL_SERVER_ERROR


@app.post("/matting/url")
async def matting_url(request: Request, model: str = Form(default=default_model, alias="model")):
    try:
        json_data = await request.json()
        image_url = json_data.get("image_url")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error parsing JSON data: {str(e)}")

    if not image_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image URL is required")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url)
            response.raise_for_status()
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to fetch image from URL: {str(e)}")

    if model not in model_paths:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid model selection")

    selected_model = model_loader.load_model(model)
    original_image_filename, image_filename, mask_filename = get_filename()
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, original_image_filename), img)

    final_img = convert_image_to_white_background(image=img)
    if final_img is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image")
    result = selected_model(final_img)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, image_filename), result[OutputKeys.OUTPUT_IMG])
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, mask_filename), result[OutputKeys.OUTPUT_IMG][:, :, 3])

    response_data = {
        "code": 0,
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
    return templates.TemplateResponse(
        "index.html", {
            "request": request,
            "default_model": default_model,
            "available_models": list(model_paths.keys())
        })


if __name__ == "__main__":
    import uvicorn

    default_bind_host = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"
    uvicorn.run(app, host=default_bind_host, port=8000)
