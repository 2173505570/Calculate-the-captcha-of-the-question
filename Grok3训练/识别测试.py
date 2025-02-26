import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from concurrent.futures import ThreadPoolExecutor
from sympy import sympify

# 设置多线程
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# 加载 TFLite 模型
interpreter = tf.lite.Interpreter(model_path='captcha_char_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# 推理函数
def predict_tflite(segments):
    predictions = []
    for segment in segments:
        # 将单个样本扩展为 (1, 28, 28, 1) 的形状
        segment = np.expand_dims(segment, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], segment)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output)
    return np.array(predictions)


# 优化预处理
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28, 28))
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return img.astype(np.float32) / 255.0


# 切分和识别
def split_and_recognize(image_path, split_points, save_dir='output'):
    img = Image.open(image_path)
    width, height = img.size
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 切分图像
    regions = [img.crop((split_points[i], 0, split_points[i + 1], height))
               for i in range(len(split_points) - 1)]

    # 多线程预处理
    with ThreadPoolExecutor() as executor:
        segments = list(executor.map(preprocess_image, regions))

    # 组合成批量输入
    segments = np.array(segments).reshape(-1, 28, 28, 1)
    predictions = predict_tflite(segments)

    # 字符映射
    char_map = {i: str(i) for i in range(10)}
    char_map.update({10: '+', 11: '-', 12: '*', 13: '/'})
    return [char_map[np.argmax(pred)] for pred in predictions]


# 构建表达式
def build_expression(recognized_chars):
    return "".join(recognized_chars)


# 主程序
image_path = r"G:\python-project\训练模型\images\0a3d34218b54969627b255f2617f8f3f.png"
split_points = [0, 31, 49, 72]
if not os.path.exists(image_path):
    raise FileNotFoundError(f"图像文件未找到: {image_path}")

# 测量时间
import time

start = time.time()
recognized_chars = split_and_recognize(image_path, split_points)
end = time.time()
print(f"推理用时: {(end - start) * 1000:.2f}ms")

expression = build_expression(recognized_chars)
print(f"识别的表达式: {expression}")

# 安全计算表达式
try:
    result = sympify(expression).evalf()
    print(f"The result of the expression '{expression}' is {result}.")
except Exception as e:
    print(f"Error evaluating the expression: {e}")