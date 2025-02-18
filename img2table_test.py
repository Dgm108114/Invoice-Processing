import os
import cv2
import yaml
from ultralytics import YOLO
from img2table.document import Image as Img2TableImage
from img2table.ocr import PaddleOCR as Img2TablePaddleOCR

pdf_path = os.path.join(os.path.dirname(__file__),"Datasets/All images1/special_cases")
config_path = os.path.join(os.path.dirname(__file__),"config/config.yaml")
output_dir = os.path.join(os.path.dirname(__file__),"Datasets/output_tables")
os.makedirs(output_dir,exist_ok=True)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

with open(config_path) as f:
    config = yaml.safe_load(f)

model = YOLO(config['model']['path'])

for idx_img,img in enumerate(os.listdir(pdf_path)):
    img2table_img = Img2TableImage(os.path.join(pdf_path,img))
    cv2_img = cv2.imread(os.path.join(pdf_path,img))

    ocr = Img2TablePaddleOCR(lang='en')
    tables = img2table_img.extract_tables(ocr=ocr,
                                          implicit_rows=True,
                                          borderless_tables=True,
                                          min_confidence=50
                                )

    for idx_table,table in enumerate(tables):
        for row in table.content.values():
            for cell in row:
                table_image_path = os.path.join(output_dir,f"table_image_{idx_img}_{idx_table}.jpg")
                temp = cv2.rectangle(cv2_img, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (255, 0, 0), 2)
                cv2.imwrite(table_image_path,temp)
                print("file saved.")



