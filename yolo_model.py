import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from itertools import product
from typing import Union
import cv2
import numpy
import numpy as np
from skimage.morphology import medial_axis
from tqdm import tqdm
from ultralytics import YOLO
import math as m
from openpyxl import load_workbook
import common

ROOT_DIR = os.path.abspath(os.getcwd())
sys.path.append(ROOT_DIR)
TEMPLATE_REPORT = os.path.join(ROOT_DIR, 'template_report.xlsx')


def predict(model_path: str, img_path: str):
    # Load a model
    model = YOLO(model_path)  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(img_path)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk
def train( dataset_dir: str, num_epochs: int, imgsz: int, model_path: str = "yolov8l-seg.pt", l2_lambda=0.0005):

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = YOLO(model_path)  # load a pretrained model (recommended for training)
    results = model.train(data=dataset_dir, epochs=num_epochs, imgsz=imgsz, patience=0, weight_decay=l2_lambda)
    print(results)

if __name__ == '__main__':
    train("datasets/wall/v4/data.yaml", 150, 640)
