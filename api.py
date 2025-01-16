# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import os
#
# app = Flask(__name__)
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
# @app.route('/recognize', methods=['POST'])
# def recognize():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
#     if file:
#         # 确保上传目录存在
#         upload_dir = 'uploads'
#         if not os.path.exists(upload_dir):
#             os.makedirs(upload_dir)
#
#         image_path = os.path.join(upload_dir, file.filename)
#         file.save(image_path)
#
#         # 使用固定的切分点
#         split_points = [0, 31, 49, 72]
#
#         recognized_chars = split_and_recognize(image_path, split_points)
#         expression = build_expression(recognized_chars)
#
#         try:
#             result = eval(expression)
#             response = {
#                 'expression': expression,
#                 'result': result
#             }
#         except Exception as e:
#             response = {'error': str(e)}
#
#         return jsonify(response)
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import os

app = FastAPI()

# 加载模型
num_model = tf.keras.models.load_model('digit_operator_recognition_model.h5')
operator_model = tf.keras.models.load_model('operator_recognition_model.h5')

# 图像预处理函数
def preprocess_image(image):
    img = image.convert('L').point(lambda p: p > 127 and 255).resize((32, 32))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 32, 32, 1)

# 去除干扰线
def remove_noise(image):
    img = np.array(image)
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    cleaned_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(cleaned_img)

# 切分图像并识别
def split_and_recognize(image, split_points):
    width, height = image.size
    recognized_chars = []

    def process_region(i, left, right):
        region = image.crop((left, 0, right, height)).convert('L')
        region = remove_noise(region).point(lambda p: p > 127 and 255)
        processed_char = preprocess_image(region)

        if i % 2 == 0:  # 偶数位置的是数字
            pred = num_model.predict(processed_char)
            return np.argmax(pred)
        else:  # 奇数位置的是运算符
            pred = operator_model.predict(processed_char)
            return np.argmax(pred)

    with ThreadPoolExecutor(max_workers=4) as executor:  # 可以根据CPU核心数调整max_workers
        futures = []
        for i in range(len(split_points) - 1):
            left = split_points[i]
            right = split_points[i + 1]
            futures.append(executor.submit(process_region, i, left, right))

        for future in futures:
            recognized_chars.append(future.result())

    return recognized_chars

# 构建表达式
def build_expression(recognized_chars):
    operators = ["+", "-", "*", "/"]
    expression = ""
    for i, char in enumerate(recognized_chars):
        if i % 2 == 0:  # 偶数位置的是数字
            expression += str(char)
        else:  # 奇数位置的是运算符
            expression += operators[char]

    return expression

@app.post("/predict")
async def recognize(file: UploadFile = File(...)):
    image = Image.open(file.file)

    # 使用固定的切分点
    split_points = [0, 31, 49, 72]

    recognized_chars = split_and_recognize(image, split_points)
    expression = build_expression(recognized_chars)

    try:
        result = eval(expression)
        response = {
            'expression': expression,
            'result': result
        }
    except Exception as e:
        response = {'error': str(e)}

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, workers=4)  # 根据服务器核心数调整workers数量