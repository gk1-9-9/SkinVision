from inference import get_model
import cv2
from gradio_client import Client, handle_file
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


API = "key"
client = Client("maxiw/Qwen2-VL-Detection")

model = get_model(model_id="acne-yolo/1")

model2path = "pathTo/jaundice.keras"

model2 = tf.keras.models.load_model(model2path)


