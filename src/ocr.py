import logging
import traceback
from paddleocr import PaddleOCR
import re
import os
import pytesseract
from PIL import Image
from io import BytesIO

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# PaddleOCR - AR - 31/05/2024
class OCRProcessor:
    def __init__(self, config):
        try:
            self.ocr = PaddleOCR(**config)
            logger.info("Initialized OCRProcessor")
        except Exception as e:
            logger.error(f"Error initializing OCRProcessor: {e}")
            logger.debug(traceback.format_exc())
            raise

    def extract_text(self, image_path):
        try:
            result = self.ocr.ocr(image_path)
            if result is None:
                return ""
            extracted_text = ' '.join([word[1][0] for line in result for word in line])
            extracted_text = re.sub(r':', ' : ', extracted_text)
            extracted_text = re.sub(r'\s+', ' ', extracted_text)
            # print("extracted_text :",extracted_text)
            return extracted_text
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            logger.debug(traceback.format_exc())
            return ""

# # Pytesseract
# class OCRProcessor:
#     def __init__(self, config):
#         try:
#             # Ensure Tesseract is installed and available
#             pytesseract.pytesseract.tesseract_cmd = config.get('tesseract_cmd', 'tesseract')
#             logger.info("Initialized OCRProcessor with Tesseract")
#         except Exception as e:
#             logger.error(f"Error initializing OCRProcessor: {e}")
#             logger.debug(traceback.format_exc())
#             raise
#
#     def extract_text(self, image_path):
#         try:
#             # image = Image.open(image_path)
#             result = pytesseract.image_to_string(image_path)
#             extracted_text = re.sub(r':', ' : ', result)
#             extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
#             # print("extracted_text :", extracted_text)
#             return extracted_text
#         except Exception as e:
#             logger.error(f"Error extracting text: {e}")
#             logger.debug(traceback.format_exc())
#             return ""

# # Pytesseract on img array
# class OCRProcessor:
#     def __init__(self, config):
#         try:
#             # Ensure Tesseract is installed and available
#             pytesseract.pytesseract.tesseract_cmd = config.get('tesseract_cmd', 'tesseract')
#             logger.info("Initialized OCRProcessor with Tesseract")
#         except Exception as e:
#             logger.error(f"Error initializing OCRProcessor: {e}")
#             logger.debug(traceback.format_exc())
#             raise
#
#     def extract_text(self, img_array):
#         try:
#             img = Image.open(BytesIO(img_array))
#             result = pytesseract.image_to_string(img)
#             extracted_text = re.sub(r':', ' : ', result)
#             extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
#             return extracted_text
#         except Exception as e:
#             logger.error(f"Error extracting text: {e}")
#             logger.debug(traceback.format_exc())
#             return ""

# import numpy as np
# # Paddle ocr on img array
# class OCRProcessor:
#     def __init__(self, config):
#         try:
#             self.ocr = PaddleOCR(**config)
#             logger.info("Initialized OCRProcessor")
#         except Exception as e:
#             logger.error(f"Error initializing OCRProcessor: {e}")
#             logger.debug(traceback.format_exc())
#             raise
#
#     def extract_text(self, img_array):
#         try:
#             img = Image.open(BytesIO(img_array))
#             img = np.array(img)
#             result = self.ocr.ocr(img)
#             if result is None:
#                 return ""
#             extracted_text = ' '.join([word[1][0] for line in result for word in line])
#             extracted_text = re.sub(r':', ' : ', extracted_text)
#             extracted_text = re.sub(r'\s+', ' ', extracted_text)
#             return extracted_text
#         except Exception as e:
#             logger.error(f"Error extracting text: {e}")
#             logger.debug(traceback.format_exc())
#             return ""
