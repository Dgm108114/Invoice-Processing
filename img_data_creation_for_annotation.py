import os
from pdf2image import convert_from_path

pdf_dir = os.path.join(os.path.dirname(__file__),  'Datasets', 'PINC Policy -Part 3')
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_dir, pdf_file)
    output_dir = os.path.join(os.path.dirname(__file__), 'Datasets', 'All images')
    os.makedirs(output_dir, exist_ok=True)

    images = convert_from_path(pdf_path, 500, poppler_path='D:/poppler-24.02.0/Library/bin')

    for idx,img in enumerate(images):
        image_path = os.path.join(output_dir, f"{pdf_file[:-4]}_{idx}.jpg")
        img.save(image_path, "JPEG")
print("All images saved.")
