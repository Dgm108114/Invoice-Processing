import os
import shutil


def rename_and_move_images(src_dir, dest_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Get a list of all files in the source directory
    files = os.listdir(src_dir)

    # Filter out only image files (optional)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]

    # Rename and move each image file
    for i, filename in enumerate(image_files):
        # Define the new name with numbering
        new_name = f"{i + 1}{os.path.splitext(filename)[1]}"

        # Define the full path for source and destination
        src_path = os.path.join(src_dir, filename)
        dest_path = os.path.join(dest_dir, new_name)

        # Move the file to the new directory with the new name
        shutil.move(src_path, dest_path)

        print(f"Moved: {src_path} -> {dest_path}")


# Example usage
src_directory = 'Datasets/All images'
dest_directory = 'Datasets/All images1'

rename_and_move_images(src_directory, dest_directory)
