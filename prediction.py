# Different try to predict - Success

# import os
# import cv2
# from ultralytics import YOLO
#
# # IMAGES_DIR = os.path.join('.', 'test','images')
# # image_path = os.path.join(IMAGES_DIR, 'VO0276521_page_1_jpg.rf.dec4387656f940ebc65b908bf91c7d20.jpg')
# image_path = "Datasets/output_images/Pedido compra - 2024-02-07T095243.891_page_1.jpg"
# image_path_out = 'Datasets/prediction'
#
# # Load an image
# frame = cv2.imread(image_path)
#
# # Load a model
# model_path = os.path.join('.', 'runs', 'detect', 'train3_e400', 'weights', 'last.pt')
# model = YOLO(model_path)  # load a custom model
#
# threshold = 0.5
#
# # Perform object detection on the image
# results = model(frame)[0]
#
# # Check if any objects were detected
# print("len(results.boxes) :",len(results.boxes))
# if len(results.boxes) > 0:
#     # Accessing detected bounding boxes
#     for result in results.boxes:
#         x1, y1, x2, y2 = result.xyxy[0]  # Extracting bounding box coordinates
#         conf = result.conf[0]  # Extracting confidence score
#         class_id = result.cls[0]  # Extracting class ID
#
#         if conf > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, model.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
# else:
#     print("No objects detected in the image.")
#
#
# # # Save the output image
# cv2.imwrite(image_path_out, frame)
#
# cv2.imshow('Object Detection', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Code to store the prediction in specific directory - AR - 26/04/24
import os
import cv2
from ultralytics import YOLO

# Load model
# # Model taken from epochs 400 (colab) - yolov8s
# model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

# # Model taken from epochs 400 (colab)- yolov8n
# model_path = os.path.join('.', 'runs', 'detect', 'train3_e400', 'weights', 'best.pt')

# Model taken from epochs 400 (colab)- yolov8n
model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'best.pt')

# # Model taken from epochs 300 (colab)- yolov8n
# model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')


model = YOLO(model_path)  # load a custom model

threshold = 0.5  # Confidence threshold for drawing bounding boxes

# Directory paths
IMAGES_DIR = "Datasets/All images1"  # Directory containing input images

PREDICTION_DIR = "Datasets/prediction_policy_output_images_PDV2_e400_v8n_preprocessed"  # Directory to save annotated images

# Ensure prediction directory exists, if not create it
os.makedirs(PREDICTION_DIR, exist_ok=True)

# Iterate over each image in the directory
for filename in os.listdir(IMAGES_DIR):
    # Path to the input image
    image_path = os.path.join(IMAGES_DIR, filename)

    # Load the image
    frame = cv2.imread(image_path)

    # Pre-process the image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # Perform object detection on the image
    results = model(frame)[0]

    # Check if any objects were detected
    if len(results.boxes) > 0:
        # Accessing detected bounding boxes
        for result in results.boxes:
            x1, y1, x2, y2 = result.xyxy[0]  # Extracting bounding box coordinates
            conf = result.conf[0]  # Extracting confidence score
            class_id = result.cls[0]  # Extracting class ID

            if conf > threshold:
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, model.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        print(f"No objects detected in the image: {filename}")

    # Save the annotated image in the prediction directory
    output_path = os.path.join(PREDICTION_DIR, f"{os.path.splitext(filename)[0]}_annotated.jpg")
    cv2.imwrite(output_path, frame)

print("Annotation completed. Annotated images saved in:", PREDICTION_DIR)



# # Code to store the prediction for single image - AR - 03/06/24
# import os
# import cv2
# from ultralytics import YOLO
#
#
# # Load a model
# # Model taken from epochs 400 (colab) - 1000 images
# model_path = os.path.join('.', 'runs', 'detect', 'colab_train_e400_i1000', 'best.pt')
#
#
# model = YOLO(model_path)  # load a custom model
#
# threshold = 0.5  # Confidence threshold for drawing bounding boxes
#
# # Directory paths
# IMAGES_DIR = "Datasets/PINC Policy -Part 2"  # Directory containing input images
#
# filename = "638411793159454123GJ12AY6902 ICICI Goods Carrying Vehicles Package Policy.pdf"
#
# PREDICTION_DIR = IMAGES_DIR  # Directory to save annotated images
#
# # Ensure prediction directory exists, if not create it
# os.makedirs(PREDICTION_DIR, exist_ok=True)
#
#
# # Path to the input image
# image_path = os.path.join(IMAGES_DIR, filename)
#
# # Load the image
# frame = cv2.imread(image_path)
#
# # Perform object detection on the image
# results = model(frame)[0]
#
# # Check if any objects were detected
# if len(results.boxes) > 0:
#     # Accessing detected bounding boxes
#     for result in results.boxes:
#         x1, y1, x2, y2 = result.xyxy[0]  # Extracting bounding box coordinates
#         conf = result.conf[0]  # Extracting confidence score
#         class_id = result.cls[0]  # Extracting class ID
#
#         if conf > threshold:
#             # Draw bounding box and label
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, model.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
# else:
#     print(f"No objects detected in the image: {filename}")
#
# # Save the annotated image in the prediction directory
# output_path = os.path.join(PREDICTION_DIR, f"{os.path.splitext(filename)[0]}_annotated.jpg")
# cv2.imwrite(output_path, frame)
#
#
# print("Annotation completed. Annotated images saved in:", PREDICTION_DIR)
# # Code to store the prediction for single image - AR - 03/06/24



# Testing
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
            # cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            if class_name in ['policy-vehicle-table', 'vehicle-table']:
                special_image_path = os.path.join("Datasets/All images1", f"{pdf_file[:-4]}_special_image_{idx}.jpg")

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

                    output_path = os.path.join(output_dir, f"table_Text.xlsx")
                    with pd.ExcelWriter(output_path) as writer:
                        df.to_excel(writer, sheet_name="table_Text", index=False)
                    # df.to_excel("extracted_text.xlsx", excel_writer=output_dir, index=False)

                    # Initialize an empty string to store the formatted text
                    formatted_text = ""

                    # Iterate through the DataFrame columns and their values
                    for column in df.columns:
                        value = df[column].iloc[0]
                        formatted_text += f"{column} {value} ,"

                    entities, _ = ner_processor.apply_custom_ner_model(df.to_string())
                    paragraph_entities.extend(entities)
                    # Print the formatted text
                    # print(formatted_text.strip())
                    # print(df.to_string())

                # cropped_image_pil.save(special_image_path)
                logger.info(f"Special case detected, saved image to: {special_image_path}")
                continue  # Skip OCR for these classes

            elif class_name in ['IDV', 'NCB', 'company-name', 'product-name', 'total-premium', 'own-damage']:

                text = ocr_processor.extract_text(cropped_image)  # Implement OCR logic
                # print("text_class_name : ", text)
                regex_result = apply_regex(text, class_name)  # Implement regex logic
                # paragraph_entities.append({'Entity': regex_result, 'Label': class_name})

            else:
                text = ocr_processor.extract_text(cropped_image)  # Implement OCR logic
                output_txt += f"{text} <////> "
                entities, _ = ner_processor.apply_custom_ner_model(text)
                paragraph_entities.extend(entities)

        return paragraph_entities, table_data, output_txt

    except Exception as e:
        logger.error(f"Error processing page {idx+1}: {e}")
        logger.debug(traceback.format_exc())
        return [], []