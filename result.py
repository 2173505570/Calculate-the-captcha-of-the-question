# import cv2
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import os
#
# # 加载模型
# num_model = tf.keras.models.load_model('num_recognition_model.h5')
# operator_model = tf.keras.models.load_model('operator_recognition_model.h5')
#
# # 图像预处理函数
# def preprocess_image(image):
#     img = image.convert('L')  # 转换为灰度图
#     img = img.point(lambda p: p > 127 and 255)  # 二值化处理
#     img = img.resize((32, 32))  # 调整尺寸
#     img_array = np.array(img) / 255.0  # 归一化
#     return img_array.reshape(1, 32, 32, 1)  # 调整形状以匹配模型输入
#
# # 去除干扰线
# def remove_noise(image):
#     img = np.array(image)
#     _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#     kernel = np.ones((2, 2), np.uint8)
#     cleaned_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
#     return Image.fromarray(cleaned_img)
#
# # 切分图像并识别
# def split_and_recognize(image_path, split_points, save_dir='output'):
#     img = Image.open(image_path)
#     width, height = img.size
#     recognized_chars = []
#
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     for i in range(len(split_points) - 1):
#         left = split_points[i]
#         right = split_points[i + 1]
#         region = img.crop((left, 0, right, height))  # 裁剪图像区域
#         region = region.convert('L')  # 转换为灰度图
#         region = remove_noise(region)  # 去除干扰线
#         region = region.point(lambda p: p > 127 and 255)  # 再次二值化处理
#         region.save(os.path.join(save_dir, f'region_{i}.png'))  # 保存裁剪区域
#
#         processed_char = preprocess_image(region)
#
#         if i % 2 == 0:  # 偶数位置的是数字
#             pred = num_model.predict(processed_char)
#             recognized_chars.append(np.argmax(pred))
#         else:  # 奇数位置的是运算符
#             pred = operator_model.predict(processed_char)
#             recognized_chars.append(np.argmax(pred))
#
#     return recognized_chars
#
# # 构建表达式
# def build_expression(recognized_chars):
#     expression = ""
#     for i, char in enumerate(recognized_chars):
#         if i % 2 == 0:  # 偶数位置的是数字
#             expression += str(char)
#         else:  # 奇数位置的是运算符
#             expression += ["+", "-", "*", "/"][char]  # 根据实际情况替换运算符
#
#     return expression
#
# # 主程序
# image_path = 'G:/python-project/训练模型/images/0a3d34218b54969627b255f2617f8f3f.png'
# split_points = [0, 31, 49, 72]  # 根据需求调整这些坐标
#
# recognized_chars = split_and_recognize(image_path, split_points)
# expression = build_expression(recognized_chars)
#
# try:
#     result = eval(expression)
#     print(f"The result of the expression '{expression}' is {result}.")
# except Exception as e:
#     print(f"Error evaluating the expression: {e}")
# import cv2
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import os
#
# # 加载单一模型（同时识别数字和运算符）
# model = tf.keras.models.load_model('digit_operator_recognition_model.h5')  # 一个模型同时处理数字和运算符
#
# # 标签映射（根据训练时的设置）
# label_to_char = {
#     0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',  # 数字
#     10: '+', 11: '-', 12: '*', 13: '/'  # 运算符
# }
#
#
# # 图像预处理函数
# def preprocess_image(image):
#     """
#     对单个字符图片进行预处理：灰度化、二值化、调整大小、归一化。
#     """
#     img = image.convert('L')  # 转换为灰度图
#     img = img.point(lambda p: p > 127 and 255)  # 二值化
#     img = img.resize((32, 32))  # 调整到模型输入要求的大小
#     img_array = np.array(img) / 255.0  # 归一化
#     return img_array.reshape(1, 32, 32, 1)  # 调整形状以匹配模型输入
#
#
# # 去除干扰线
# def remove_noise(image):
#     """
#     对整张图片去除噪声（如干扰线）。
#     """
#     img = np.array(image)
#     _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#     kernel = np.ones((2, 2), np.uint8)
#     cleaned_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
#     return Image.fromarray(cleaned_img)
#
#
# # 切分图像并识别
# def split_and_recognize(image_path, split_points, save_dir='output'):
#     """
#     切分输入图片并识别每个字符。
#     """
#     img = Image.open(image_path)
#     width, height = img.size
#     recognized_chars = []
#
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     for i in range(len(split_points) - 1):
#         left = split_points[i]
#         right = split_points[i + 1]
#         region = img.crop((left, 0, right, height))  # 裁剪图像区域
#         region = region.convert('L')  # 转换为灰度图
#         region = remove_noise(region)  # 去除干扰线
#         region = region.point(lambda p: p > 127 and 255)  # 再次二值化处理
#         region.save(os.path.join(save_dir, f'region_{i}.png'))  # 保存裁剪区域
#
#         # 对字符图像进行预测
#         processed_char = preprocess_image(region)
#         pred = model.predict(processed_char)  # 使用单一模型进行预测
#         recognized_char = label_to_char[np.argmax(pred)]
#         recognized_chars.append(recognized_char)
#
#     return recognized_chars
#
#
# # 构建表达式
# def build_expression(recognized_chars):
#     """
#     将识别的字符列表拼接成数学表达式。
#     """
#     return ''.join(recognized_chars)
#
#
# # 主程序
# if __name__ == "__main__":
#     # 输入图片路径
#     image_path = ""
#
#     # 切分点（需要根据图像调整）
#     split_points = [0, 31, 49, 72]  # 根据需求调整切分坐标
#
#     # 识别和计算
#     recognized_chars = split_and_recognize(image_path, split_points)
#     expression = build_expression(recognized_chars)
#
#     try:
#         result = eval(expression)  # 计算表达式的值
#         print(f"The result of the expression '{expression}' is {result}.")
#     except Exception as e:
#         print(f"Error evaluating the expression: {e}")

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# 加载模型
model = tf.keras.models.load_model('digit_operator_recognition_model.h5')  # 一个模型同时处理数字和运算符

# 图像预处理函数
def preprocess_image(image):
    img = image.convert('L')  # 转换为灰度图
    img = img.point(lambda p: p > 127 and 255)  # 二值化处理
    img = img.resize((32, 32))  # 调整尺寸
    img_array = np.array(img) / 255.0  # 归一化
    return img_array

# 去除干扰线
def remove_noise(image):
    img = np.array(image)
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    cleaned_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(cleaned_img)

# 切分图像并识别
def split_and_recognize(image_path, split_points, save_dir='output'):
    img = Image.open(image_path)
    width, height = img.size
    recognized_chars = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    segments = []
    for i in range(len(split_points) - 1):
        left = split_points[i]
        right = split_points[i + 1]
        region = img.crop((left, 0, right, height))  # 裁剪图像区域
        region = region.convert('L')  # 转换为灰度图
        region = remove_noise(region)  # 去除干扰线
        region = region.point(lambda p: p > 127 and 255)  # 再次二值化处理
        region.save(os.path.join(save_dir, f'region_{i}.png'))  # 保存裁剪区域

        # 预处理
        processed_char = preprocess_image(region)
        segments.append(processed_char)  # 批量存储

    # 批量预测
    segments = np.array(segments).reshape(-1, 32, 32, 1)  # 适配输入
    predictions = model.predict(segments)

    # 分类映射
    for i, pred in enumerate(predictions):
        if i % 2 == 0:  # 偶数位置的是数字
            recognized_chars.append(np.argmax(pred))
        else:  # 奇数位置的是运算符
            recognized_chars.append(np.argmax(pred))

    return recognized_chars

# 构建表达式
def build_expression(recognized_chars):
    expression = ""
    for i, char in enumerate(recognized_chars):
        if i % 2 == 0:  # 偶数位置的是数字
            expression += str(int(char))  # 使用 int() 去掉前导零
        else:  # 奇数位置的是运算符
            expression += ["+", "-", "*", "/"][char]  # 根据模型的运算符类别
    return expression

# 主程序
image_path = ""
split_points = [0, 31, 49, 72]  # 根据需求调整这些坐标

recognized_chars = split_and_recognize(image_path, split_points)
expression = build_expression(recognized_chars)

try:
    result = eval(expression)
    print(f"The result of the expression '{expression}' is {result}.")
except Exception as e:
    print(f"Error evaluating the expression: {e}")
