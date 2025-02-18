import os
import logging
import traceback
from ultralytics import YOLO
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Detector:
    def __init__(self, model_path, threshold, class_names):
        try:
            self.model = YOLO(model_path)  # Load the custom model
            self.threshold = threshold
            self.class_names = class_names
            logger.info("Initialized Detector")
            logger.info(f"Initialized Detector with model path: {model_path}")
        except Exception as e:
            logger.error(f"Error initializing Detector: {e}")
            logger.debug(traceback.format_exc())
            raise

    def detect(self, image):
        try:
            results = self.model(image)
            boxes = []
            class_ids = []

            for result in results:
                for det in result.boxes:
                    if det.conf.item() > self.threshold:  # Access the confidence score using .item()
                        boxes.append((int(det.xyxy[0][0].item()), int(det.xyxy[0][1].item()), int(det.xyxy[0][2].item()), int(det.xyxy[0][3].item())))
                        class_ids.append(int(det.cls.item()))
            return boxes, class_ids
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            logger.debug(traceback.format_exc())
            return [], []

    def get_class_name(self, class_id):
        try:
            return self.class_names[class_id]
        except IndexError:
            logger.error(f"Invalid class ID: {class_id}")
            return "Unknown"
