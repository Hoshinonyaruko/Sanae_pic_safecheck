from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io
import time
import os

class PredictionSubResult(BaseModel):
    yellow: float
    green: float

class PredictionResult(BaseModel):
    result: PredictionSubResult

# Initialize image counter
image_counter = 0

app = FastAPI()

def config_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_model(SAVE_DIR):
    with open(os.path.join(SAVE_DIR, "model.json"), 'r', encoding='utf-8') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(SAVE_DIR, "model.h5"))
    return model

# Configuration and Model Initialization
config_gpu()
SAVE_DIR = "E:\\模型\\my_model-zaomiaoweigui\\"
model = load_model(SAVE_DIR)

@app.post("/upload/", response_model=PredictionResult)
async def upload_image(file: UploadFile = File(...)):
    global image_counter
    image_counter += 1

    start_time = time.time()

    img_size = (300, 300)
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    if image.mode == 'L':  # Check if the image is in grayscale
        image = image.convert('RGB')

    image = image.resize(img_size)
    x = np.array(image)[np.newaxis]
    x = preprocess_input(x.astype(float))
    input_shape = model.input_shape

    try:
        pred = model.predict_on_batch(x)[0]
    except Exception as e:
        return {"error": str(e)}

    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000

    print(f"Elapsed time: {elapsed_time:.2f} ms")
    print(f"Total images processed: {image_counter}")  # 输出总共处理过的图片数量

    if float(pred[0]) > 0.5:
        save_image(image, file.filename, float(pred[0]), float(pred[1]))
    
    sub_result = PredictionSubResult(yellow=float(pred[0]), green=float(pred[1]))
    result = PredictionResult(result=sub_result)

    print(f"API result: {result}")  # 输出API结果
    return result

def save_image(image, filename, yellow_value, green_value):
    yellow_folder = "./yellow"
    if not os.path.exists(yellow_folder):
        os.makedirs(yellow_folder)
    formatted_yellow_value = "{:.4f}".format(yellow_value)
    formatted_green_value = "{:.4f}".format(green_value)
    file_name = f"{filename}-y{formatted_yellow_value}-g{formatted_green_value}-{time.strftime('%Y%m%d%H%M%S')}.png"
    file_path = os.path.join(yellow_folder, file_name)
    image.save(file_path)

# 使用命令运行：python -m uvicorn main:app --host 0.0.0.0 --port 5003
