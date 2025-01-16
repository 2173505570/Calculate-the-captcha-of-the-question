import cv2
import numpy as np
from PIL import Image
import ddddocr
import os
import io  # 导入 io 模块

def preprocess_and_split_image(image_path, output_folder1, output_folder2, index):
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return

    # 检查文件大小，确保文件不是空的
    if os.path.getsize(image_path) == 0:
        print(f"文件为空: {image_path}")
        return

    # 尝试使用 OpenCV 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图
    if img is None:
        try:
            # 如果 OpenCV 读取失败，尝试使用 Pillow 读取
            pil_img = Image.open(image_path).convert('L')  # 转换为灰度图
            img = np.array(pil_img)
        except Exception as e:
            print(f"无法读取图像: {image_path}, 错误信息: {e}")
            return

    # 阈值处理（二值化）
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 去除干扰线
    kernel = np.ones((1, 1), np.uint8)
    cleaned_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # 转换回PIL图像以便于裁剪
    pil_img = Image.fromarray(cleaned_img)

    # 初始化OCR实例
    ocr = ddddocr.DdddOcr()

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)

    # 切分图像并识别
    split_points = [0, 31, 49, 72]  # 根据需求调整这些坐标
    for i in range(3):  # 只切分前三段
        left = split_points[i]
        right = split_points[i + 1]
        cropped_img = pil_img.crop((left, 0, right, pil_img.height))

        # 将PIL图像转换为字节数据
        with io.BytesIO() as buffer:
            cropped_img.save(buffer, format='PNG')
            byte_data = buffer.getvalue()

        # OCR识别
        result = ocr.classification(byte_data)

        # 确保结果是有效的文件名
        if not result:
            result = 'unknown'
        else:
            # 替换非法字符
            result = ''.join(c for c in result if c.isalnum())

        # 选择合适的输出文件夹
        if i == 1:
            output_folder = output_folder2
        else:
            output_folder = output_folder1

        # 保存切分后的图像，并使用识别结果作为文件名的一部分
        output_file_name = f'{result}_{index}_{i}.png'
        output_file_path = os.path.join(output_folder, output_file_name)
        print(f"保存文件: {output_file_path}")
        cropped_img.save(output_file_path)

def process_images_in_folder(input_folder, output_folder1, output_folder2):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)

    # 获取输入文件夹中的所有图片
    images = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 遍历所有图片
    for index, filename in enumerate(images):
        image_path = os.path.join(input_folder, filename)
        print(f"正在处理第 {index + 1} 张图片: {filename}")
        preprocess_and_split_image(image_path, output_folder1, output_folder2, index)

# 使用示例
input_folder = 'G:\\python-project\\训练模型\\images'  # 输入文件夹路径
output_folder1 = 'G:\\python-project\\训练模型\\数字'  # 第一和第三张图片的输出文件夹
output_folder2 = 'G:\\python-project\\训练模型\\运算符'  # 第二张图片的输出文件夹
process_images_in_folder(input_folder, output_folder1, output_folder2)