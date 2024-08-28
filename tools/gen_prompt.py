import cv2
import os
import numpy as np

def read_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

def convert_to_binary(images):
    binary_images = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        binary_images.append(binary)
    return binary_images

def apply_dilation(images, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    dilated_images = [cv2.dilate(img, kernel, iterations=1) for img in images]
    return dilated_images

def save_images_to_folder(images, filenames, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for img, filename in zip(images, filenames):
        cv2.imwrite(os.path.join(output_folder, filename), img)

def process_images(input_folder, output_folder):
    images, filenames = read_images_from_folder(input_folder)
    binary_images = convert_to_binary(images)
    dilated_images = apply_dilation(binary_images)
    save_images_to_folder(dilated_images, filenames, output_folder)


if __name__=='__main__':
    # 设置输入和输出文件夹路径
    input_folder = '../distinct646/train/alpha'
    output_folder = '../distinct646/train/mask'

    # 处理图像
    process_images(input_folder, output_folder)
