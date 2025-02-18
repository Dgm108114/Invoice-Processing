import os
import concurrent.futures
import time
import logging
import yaml
import cv2
from pdf2image import convert_from_path
import pandas as pd
from PIL import Image
import traceback
import csv

from preprocess import preprocess_image
from detect import Detector
from ocr import OCRProcessor
from ner import NERProcessor
from utils import *
from concurrent.futures import ProcessPoolExecutor
from img2table.document import Image as Img2TableImage
from img2table.ocr import PaddleOCR as Img2TablePaddleOCR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def crop_image(image, box):
    x_min, y_min, x_max, y_max = box
    return image[y_min:y_max, x_min:x_max]

def process_page(idx, image, config, detector, ner_processor, output_dir, pdf_file):
    try:
        image_path = os.path.join(output_dir, f"temp_image_{idx}.jpg")
        image.save(image_path, "JPEG")
        img = cv2.imread(image_path)
        img = preprocess_image(img)

        # Instantiate PaddleOCR inside the process_page function
        ocr_processor = OCRProcessor(config['paddleocr'])

        boxes, class_ids = detector.detect(img)
        paragraph_entities = []
        table_data = []
        output_txt = ""

        for box, class_id in zip(boxes, class_ids):
            class_name = detector.get_class_name(class_id)
            cropped_image = crop_image(img, box)

            if class_name in ['policy-vehicle-table', 'vehicle-table']:
                special_image_path = os.path.join(os.path.dirname(__file__),'.',"Datasets\All images1\special_cases", f"{pdf_file[-15:-4]}_special_image_{idx}.jpg")
                print("special_image_path : ", special_image_path)

                table_image_path = os.path.join(output_dir, f"table_image_{idx}_{class_id}.jpg")
                cv2.imwrite(special_image_path, cropped_image)
                cv2.imwrite(table_image_path, cropped_image)
                img2table_img = Img2TableImage(table_image_path)
                ocr = Img2TablePaddleOCR(lang="en")
                tables = img2table_img.extract_tables(ocr=ocr,
                                      implicit_rows=True,
                                      borderless_tables=True,
                                      min_confidence=50)

                for table in tables:
                    df = pd.DataFrame(table.df)
                    # Use the first row as the header
                    df.columns = df.iloc[0]
                    df = df[1:]  # Skip the first row as it is now the header

                    # To save data extracted from table image
                    # output_path = os.path.join(output_dir, f"table_Text.xlsx")
                    # with pd.ExcelWriter(output_path) as writer:
                    #     df.to_excel(writer, sheet_name="table_Text", index=False)

                    # Initialize an empty string to store the formatted text
                    formatted_text = ""

                    # Iterate through the DataFrame columns and their values
                    for column in df.columns:
                        value = df[column].iloc[0]
                        formatted_text += f"{column} {value} ,"

                    output_txt += f"{formatted_text} <////> "

                # cropped_image_pil.save(special_image_path)
                logger.info(f"Special case detected, saved image to: {special_image_path}")

            elif class_name in ['IDV', 'NCB', 'company-name', 'product-name', 'total-premium', 'own-damage']:
                text = ocr_processor.extract_text(cropped_image)  # Implement OCR logic
                print("text_class_name : ", text)
                regex_result = apply_regex(text, class_name)  # Implement regex logic
                if regex_result:
                    paragraph_entities.append(regex_result)
                else:
                    pass
                # paragraph_entities.append({'Entity': regex_result, 'Label': class_name})

            else:
                text = ocr_processor.extract_text(cropped_image)  # Implement OCR logic
                output_txt += f"{text} <////> "

        output_txt = clean_text(output_txt)

        entities, _ = ner_processor.apply_custom_ner_model(output_txt)
        paragraph_entities.extend(entities)

        return paragraph_entities, table_data, output_txt

    except Exception as e:
        logger.error(f"Error processing page {idx+1}: {e}")
        logger.debug(traceback.format_exc())
        return [], []

def process_pdf_parallel(pdf_path, config, output_dir, pdf_file):
    try:
        detector = Detector(config['model']['path'], config['model']['threshold'], config['model']['class_names'])
        ner_processor = NERProcessor(config['paths']['ner_model'])

        images = convert_from_path(pdf_path, 300, poppler_path=config['paths']['poppler_path'])
        all_entities = []
        all_table_data = []
        output_txt_final = ""

        # Process the PDF in batches of pages
        batch_size = 5  # Adjust batch size as needed
        total_pages = len(images)

        for start_page in range(0, total_pages, batch_size):
            end_page = min(start_page + batch_size, total_pages)
            batch_images = images[start_page:end_page]

            with ProcessPoolExecutor() as executor:
                future_to_idx = {executor.submit(process_page, idx, image, config, detector, ner_processor, output_dir, pdf_file): idx for idx, image in enumerate(batch_images)}
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        entities, table_data, output_txt = future.result()
                        all_entities.extend(entities)
                        all_table_data.extend(table_data)
                        output_txt_final += output_txt
                    except Exception as exc:
                        logger.error(f"Page {idx+1} processing failed: {exc}")
                        logger.debug(traceback.format_exc())

        print("all_entities :", sorted(all_entities))

        temp_dict = {}
        for key, value in sorted(all_entities):
            if key in temp_dict:
                temp_dict[key].append(value)
            else:
                temp_dict[key] = [value]

        # Save the DataFrame to an Excel file
        entities_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in temp_dict.items()]))

        output_path = os.path.join(output_dir, f"Extracted_Text.xlsx")
        entities_df.to_excel(output_path, index=False)

        # To generate output file for all pdfs
        # with open("output_txt.txt", "a") as f:
        #     f.write(f"{output_txt_final} --- ")

        with open(os.path.join(output_dir,"output_txt.txt"), "a") as f:
            f.write(f"{output_txt_final} --- ")

        logger.info(f"Text extraction completed for {pdf_path}.")
    except Exception as e:
        logger.error(f"Error in process_pdf_parallel for {pdf_path}: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    start_time = time.time()

    try:
        # Use the correct relative path to the config file
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # pdf_dir = os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'PINC Policy-Part 1')
        pdf_dir = os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'Testing_pdfs')
        # pdf_file = "TATA AIG SANTOSH DASHRATH KHENGRE.pdf"
        # pdf_file = "6384171494750222713001_310451150_00_0001 ICICI Private Car Package Policy.pdf"

        # pdf_file = "638440486600367033MIRCELECTRONICSLTD  Reliance Private Car Package Policy- Schedule.pdf"
        # pdf_file = "131522323120029452 RELIANCE-D.pdf"
        pdf_file = "GJ12AU5791 ICICI GCV.pdf"
        # pdf_file = "TATA AIG SANTOSH DASHRATH KHENGRE.pdf"

        pdf_path = os.path.join(pdf_dir, pdf_file)

        output_dir = os.path.join(os.path.dirname(__file__), "..", "Result", f"{pdf_file[:-4]}")
        os.makedirs(output_dir, exist_ok=True)

        process_pdf_parallel(pdf_path, config, output_dir, pdf_file)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.debug(traceback.format_exc())

    end_time = time.time()
    logger.info(f"Execution Time: {end_time - start_time} seconds")
    logger.info("Excel file saved.")


# pdf_file = "638440486600367033MIRCELECTRONICSLTD  Reliance Private Car Package Policy- Schedule.pdf"

# # To apply on pdf directory
# if __name__ == "__main__":
#     start_time = time.time()
#     try:
#         # Use the correct relative path to the config file
#         config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
#         with open(config_path) as f:
#             config = yaml.safe_load(f)
#
#         pdf_dir = os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'All PDF')
#         pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
#
#         for pdf_file in pdf_files:
#             pdf_path = os.path.join(pdf_dir, pdf_file)
#             output_dir = os.path.join(os.path.dirname(__file__), "..", "Result","All PDF", f"{pdf_file[:-4]}")
#             os.makedirs(output_dir, exist_ok=True)
#             process_pdf_parallel(pdf_path, config, output_dir, pdf_file)
#
#     except Exception as e:
#         logger.error(f"Error in main execution: {e}")
#         logger.debug(traceback.format_exc())
#
#     end_time = time.time()
#     logger.info(f"Total Execution Time: {end_time - start_time} seconds")
#     logger.info("All PDFs processed.")


# Before applying Batch processing
# def process_pdf_parallel(pdf_path, config, output_dir, pdf_file):
#     try:
#         detector = Detector(config['model']['path'], config['model']['threshold'], config['model']['class_names'])
#         ner_processor = NERProcessor(config['paths']['ner_model'])
#
#         images = convert_from_path(pdf_path, 300, poppler_path=config['paths']['poppler_path'])
#         all_entities = []
#         all_table_data = []
#         output_txt_final = ""
#
#         with ProcessPoolExecutor() as executor:
#             future_to_idx = {executor.submit(process_page, idx, image, config, detector, ner_processor, output_dir, pdf_file): idx for idx, image in enumerate(images)}
#             for future in concurrent.futures.as_completed(future_to_idx):
#                 idx = future_to_idx[future]
#                 try:
#                     entities, table_data, output_txt = future.result()
#                     all_entities.extend(entities)
#                     all_table_data.extend(table_data)
#                     output_txt_final += output_txt
#                 except Exception as exc:
#                     logger.error(f"Page {idx+1} processing failed: {exc}")
#                     logger.debug(traceback.format_exc())
#
#         print("all_entities :",all_entities)
#
#         entities_df = pd.DataFrame(all_entities, columns=['Entity', 'Label'])
#         pivoted_df = entities_df.pivot_table(index=entities_df.index // len(all_entities), columns='Label', values='Entity', aggfunc='first')
#
#         output_path = os.path.join(output_dir, f"Extracted_Text.xlsx")
#         with pd.ExcelWriter(output_path) as writer:
#             pivoted_df.to_excel(writer, sheet_name="Extracted_Text", index=False)
#             for i, table_df in enumerate(all_table_data):
#                 table_df.to_excel(writer, sheet_name=f'Table_{i+1}', index=False)
#
#         # output_path_txt = os.path.join(output_dir, "Extracted_Text.txt")
#         # os.makedirs(output_path_txt,exist_ok=True)
#
#         # with open(output_path_txt,"a") as f:
#         #     f.write(f"{output_txt_final}")
#
#         with open("output_txt.txt","a") as f:
#             f.write(f"{output_txt_final} --- ")
#
#         logger.info(f"Text extraction completed for {pdf_path}.")
#     except Exception as e:
#         logger.error(f"Error in process_pdf_parallel for {pdf_path}: {e}")
#         logger.debug(traceback.format_exc())
