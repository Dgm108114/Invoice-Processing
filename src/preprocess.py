import cv2
import numpy as np
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def preprocess_image(image):
    try:
        logger.info("Starting image preprocessing")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is not None:
            angles = [np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi for line in lines for x1, y1, x2, y2 in line]
            median_angle = np.median(angles)
            if np.abs(median_angle) > 10:
                logger.info(f"Rotating image by {median_angle} degrees")
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE if median_angle < -45 else cv2.ROTATE_90_COUNTERCLOCKWISE)

        return image
    except Exception as e:
        logger.error(f"Error in preprocess_image: {e}")
        logger.debug(traceback.format_exc())
        return image

