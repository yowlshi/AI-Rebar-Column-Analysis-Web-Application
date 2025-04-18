# app.py

# RebarVista - Rebar Detection and Volume Calculation Application

import os
import sys
import cv2
import torch
import warnings
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import base64
import copy

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

app = Flask(__name__)
CORS(app)

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

MODEL_CONFIG_PATH = "C:/Users/angelo/Downloads/Angelo/Rebar web app/model/mask_rcnn_R_101_FPN_3x.yaml"
MODEL_WEIGHTS_PATH = "C:/Users/angelo/Downloads/Angelo/Rebar web app/model/model_final.pth"

SCALE_FACTOR_CM_PER_PIXEL = 0.1
WIDTH_OFFSET_CM = 45.0

print("----- RebarVista DEBUG INFO -----")
print("Current Working Directory:", os.getcwd())
print("Model Config Path:", MODEL_CONFIG_PATH)
print("Model Weights Path:", MODEL_WEIGHTS_PATH)
print("Config Exists:", os.path.isfile(MODEL_CONFIG_PATH))
print("Weights Exist:", os.path.isfile(MODEL_WEIGHTS_PATH))
print("----------------------------------")

if not os.path.isfile(MODEL_CONFIG_PATH):
    print(f"Configuration file not found: {MODEL_CONFIG_PATH}")
    sys.exit(1)

if not os.path.isfile(MODEL_WEIGHTS_PATH):
    print(f"Model weights not found: {MODEL_WEIGHTS_PATH}")
    sys.exit(1)

cfg = get_cfg()
try:
    cfg.merge_from_file(MODEL_CONFIG_PATH)
    print("Configuration file loaded successfully.")
except FileNotFoundError:
    print("Configuration file could not be loaded. File not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading configuration file: {e}")
    sys.exit(1)

cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.DEVICE = "cpu"

try:
    predictor = DefaultPredictor(cfg)
    print("Predictor initialized successfully.")
except Exception as e:
    print(f"Error initializing predictor: {e}")
    sys.exit(1)

def calculate_volume(bbox, scale_factor, width_offset_cm):
    x1, y1, x2, y2 = bbox
    width_pixels = x2 - x1
    length_pixels = y2 - y1
    width_cm = width_pixels * scale_factor
    length_cm = length_pixels * scale_factor
    height_cm = width_cm
    adjusted_width_cm = width_cm + width_offset_cm
    volume_cc = length_cm * adjusted_width_cm * height_cm
    return volume_cc, width_cm, length_cm, height_cm

def add_segment_labels(image, segments):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for segment in segments:
        i, vol, width, length_cm, height_cm, bbox = segment
        label = f"Segment {i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1

        x1, y1, _, _ = bbox.astype(int)
        padding = 5
        x_text = max(x1, 0) + padding
        y_text = max(y1, 0) + padding

        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(image_bgr, (x_text, y_text - text_height - 5), (x_text + text_width + 5, y_text + 5), (0, 0, 0), cv2.FILLED)
        cv2.putText(image_bgr, label, (x_text, y_text), font, font_scale, color, thickness, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        frame = np.array(img)
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")
        if len(instances) == 0:
            return jsonify({'message': 'No rebar detected', 'segments': [], 'total_volume': 0, 'image': None})

        if not instances.has("pred_masks"):
            return jsonify({'error': 'The model did not return any masks.'}), 500

        instances_wo_labels = copy.deepcopy(instances)
        for field in ["scores", "pred_classes"]:
            if instances_wo_labels.has(field):
                instances_wo_labels.remove(field)

        v = Visualizer(frame[:, :, ::-1],
                       scale=1.0,
                       instance_mode=ColorMode.SEGMENTATION)
        out = v.draw_instance_predictions(instances_wo_labels)
        segmented_img = out.get_image()[:, :, ::-1]

        boxes = instances.pred_boxes.tensor.numpy()
        segments = []
        for i, bbox in enumerate(boxes, start=1):
            vol, width_cm, length_cm, height_cm = calculate_volume(bbox, SCALE_FACTOR_CM_PER_PIXEL, WIDTH_OFFSET_CM)
            segments.append((i, vol, width_cm, length_cm, height_cm, bbox))

        segmented_img_with_labels = add_segment_labels(segmented_img, segments)

        # Change the formula to sum up volumes using a loop:
        total_volume = 0
        for seg in segments:
            total_volume += seg[1]  # seg[1] is the volume

        # Prepare segments data for JSON
        segments_data = []
        for (i, vol, width, length_cm, height_cm, _) in segments:
            segments_data.append({
                'segment_no': i,
                'volume': vol,
                'width': width,
                'length': length_cm,
                'height': height_cm
            })

        pil_image = Image.fromarray(segmented_img_with_labels)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'message': 'Detection successful',
            'segments': segments_data,
            'total_volume': total_volume,
            'image': img_str
        })

    except Exception as e:
        print(f"Exception during image processing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/processed_images/<filename>')
def send_processed_image(filename):
    return send_from_directory('static/processed_images', filename)

if __name__ == "__main__":
    app.run(debug=True)
