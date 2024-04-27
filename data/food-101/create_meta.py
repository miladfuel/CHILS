import os
import json
from PIL import Image

def rename_images_sequentially(image_dir):
    """ Renames images in each subdirectory of image_dir to a unique sequential name across all folders. """
    count = 1  # Start numbering from 1
    for subdir in sorted(os.listdir(image_dir)):
        subdir_path = os.path.join(image_dir, subdir)
        if os.path.isdir(subdir_path):
            images = sorted([img for img in os.listdir(subdir_path) if img.endswith(('.png', '.jpg', '.jpeg'))])
            for img in images:
                old_path = os.path.join(subdir_path, img)
                new_path = os.path.join(subdir_path, f"{count:02d}.jpg")  # Format as 01.jpg, 02.jpg, etc.
                os.rename(old_path, new_path)
                count += 1  # Increment the counter for the next image

def create_classes_and_labels_files(image_dir, classes_file, labels_file):
    categories = sorted(os.listdir(image_dir))
    with open(classes_file, 'w') as cf, open(labels_file, 'w') as lf:
        for category in categories:
            cf.write(category + '\n')
            formatted_name = ' '.join(word.capitalize() for word in category.split('_'))
            lf.write(formatted_name + '\n')

def ensure_rgb(image_path):
    """ Ensures the image is in RGB format """
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
            img.save(image_path)  # Overwrite the original image with RGB format if needed

def create_dataset_json(data_dir, output_file, include_extension=False):
    dataset = {}
    for category in sorted(os.listdir(data_dir)):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            if include_extension:
                images = [os.path.join(category, img) for img in sorted(os.listdir(category_path)) if img.endswith('.jpg')]
            else:
                images = [os.path.join(category, img.split('.')[0]) for img in sorted(os.listdir(category_path)) if img.endswith('.jpg')]
            dataset[category] = images
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)

def create_dataset_txt(data_dir, output_file):
    with open(output_file, 'w') as file:
        for category in sorted(os.listdir(data_dir)):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                images = sorted(os.listdir(category_path))
                for img in images:
                    if img.endswith('.jpg'):
                        file.write(f"{category}/{img.split('.')[0]}\n")

def main():
    current_dir = os.getcwd()  # Get the current working directory
    data_dir = os.path.join(current_dir, 'images')  # Adjust this if your images are stored differently
    rename_images_sequentially(data_dir)  # Rename images to ensure uniqueness

    meta_dir = os.path.join(current_dir, 'meta')
    os.makedirs(meta_dir, exist_ok=True)  # Ensure the 'meta' directory exists

    classes_file = os.path.join(meta_dir, 'classes.txt')
    labels_file = os.path.join(meta_dir, 'labels.txt')
    train_json = os.path.join(meta_dir, 'train.json')
    test_json = os.path.join(meta_dir, 'test.json')
    train_txt = os.path.join(meta_dir, 'train.txt')
    test_txt = os.path.join(meta_dir, 'test.txt')

    create_classes_and_labels_files(data_dir, classes_file, labels_file)
    create_dataset_json(data_dir, train_json, include_extension=True)
    create_dataset_json(data_dir, test_json, include_extension=False)
    create_dataset_txt(data_dir, train_txt)
    create_dataset_txt(data_dir, test_txt)

if __name__ == '__main__':
    main()

