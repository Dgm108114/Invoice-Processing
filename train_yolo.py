import os

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # pretrained model
# model = YOLO("/content/drive/MyDrive/Yolo_results/train2/weights/last.pt")  # build a new model from scratch


# Specify the save directory for training runs
# Project = '/content/drive/MyDrive/Yolo_results'
# os.makedirs(Project, exist_ok=True)

# ## Use the model - Trial 3 (train3_e400)
# results = model.train(data="Datasets/Roboflow Dataset PDv2 v1 308 actual img/data.yaml", epochs=400, imgsz=640, augment=True, workers=0)  # train the model

# ## Use the model - Trial 4 (train3_e400)
results = model.train(data="Datasets/Roboflow Dataset PDv2 v2 308 actual img/data.yaml", epochs=400, imgsz=640, batch=32, augment=True, workers=0)  # train the model

## Resume the model - Trial 3 (train)
# results = model.train(data=os.path.join(ROOT_DIR, "google_colab_config.yaml"), epochs=500, imgsz=640, batch=32, augment=True, resume=True, project=Project)  # train the model

# results.save_dir(save_dir='/content/sample_data')