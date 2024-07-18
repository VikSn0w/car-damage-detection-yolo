import os
import random

from PIL import Image

import cv2
import numpy as np
from ultralytics import YOLO

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def predict(model_path: str, img_path: str,  type="object-segmentation"):
    if type == "object-segmentation":
        # Load trained segmentation model
        img = cv2.imread(img_path)
        model = YOLO(model_path)

        # Run inference (prediction) on images
        results = model.predict(img_path)

        keys = model.names
        keys = keys.keys()
        colors = dict()
        for key in keys:
            colors[key] = get_random_color()

        print(colors)

        if len(results) > 0:
            result = results[0]
            for i, mask in enumerate(result.masks.data):
                mask_app = mask.cpu().detach().numpy()
                gray_image = cv2.convertScaleAbs(mask_app)

                cls = result.boxes.cls[i]
                class_name = int(cls)
                print(class_name)


                #Apply a threshold to get a binary image (optional, depending on your use case)
                _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                overlay = img.copy()
                cv2.drawContours(overlay, contours, -1, colors[class_name], -1)  # Filled contour
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)  # Blend overlay with the original image

                cv2.drawContours(img, contours, -1, colors[class_name], 2)  # Contour outline


                x, y, w, h = cv2.boundingRect(contours[0])

                # Put class name on top
                cv2.putText(img, (model.names[class_name]), (x, y-h), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[class_name], 2)

        return img

    elif type == "object-detection":
        # Load trained segmentation model
        img = cv2.imread(img_path)
        model = YOLO(model_path)

        # Run inference (prediction) on images
        results = model.predict(img_path)

        keys = model.names
        keys = keys.keys()
        colors = dict()
        for key in keys:
            colors[key] = get_random_color()

        for result in results:
            for box in result.boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten().tolist())
                # Get the class label and confidence score
                cls = int(box.cls)
                conf = box.conf.item()

                # Draw the bounding box on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls], 2)
                # Put the class label and confidence score
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  colors[cls], 2)
        return img
    else:
        print("ERRORE: tipo di predict non specificato o non valido")


def train( dataset_dir: str, num_epochs: int, imgsz: int, type = "object-segmentation", l2_lambda=0.0005, patience=0, batch=16):
    model_path = ""
    if type == "object-segmentation":
        model_path = "yolov8l-seg.pt"
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
    elif type == "object-detection":
        model_path = "yolov8l.pt"
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

    if model_path != "":
        model = YOLO(model_path)  # load a pretrained model (recommended for training)
        results = model.train(data=dataset_dir, epochs=num_epochs, imgsz=imgsz, patience=patience, batch=batch, weight_decay=l2_lambda)
        print(results)
    else:
        print("ERRORE: tipo di training non specificato o non valido")

if __name__ == '__main__':
    model_type = "object-segmentation"
    #train("dataset/object-detection/cats_dogs/data.yaml", 100, 640, type=model_type)
    img = predict("runs/segment/train2/weights/best.pt", "test.jpg", type=model_type)
    cv2.imshow("img", img)
    cv2.waitKey(0)

