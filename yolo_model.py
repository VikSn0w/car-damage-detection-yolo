import os
from PIL import Image

import cv2
import numpy as np
from ultralytics import YOLO


def predict(model_path: str, img_path: str):
    # Load trained segmentation model
    img = cv2.imread(img_path)
    model = YOLO(model_path)

    # Run inference (prediction) on images
    results = model.predict(img_path)
    result = results[0].masks.data
    for mask in result:
        mask_app = mask.cpu().detach().numpy()
        gray_image = cv2.convertScaleAbs(mask_app)

        # Apply a threshold to get a binary image (optional, depending on your use case)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours on the image

        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    return img



def train( dataset_dir: str, num_epochs: int, imgsz: int, model_path: str = "yolov8l-seg.pt", l2_lambda=0.0005):

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = YOLO(model_path)  # load a pretrained model (recommended for training)
    results = model.train(data=dataset_dir, epochs=num_epochs, imgsz=imgsz, patience=0, batch=16, weight_decay=l2_lambda)
    print(results)

if __name__ == '__main__':
    #train("dataset/data.yaml", 100, 640)
    img = predict("runs/segment/train2/weights/best.pt", "test.jpg")
    cv2.imshow("img", img)
    cv2.waitKey(0)

