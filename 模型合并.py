# -*- coding = utf-8 -*-
# @Time :2024/11/15 22:06
# @Author hai
# @File : 模型合并.py
# @Software: PyCharm
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# 加载 Keras 模型
model = tf.keras.models.load_model('digit_operator_recognition_model.h5')

# 准备数据（模拟数据示例，你可以换成实际的图片数据）
# 假设 `X_data` 是图像数据, `y_data` 是标签
X_data = np.random.rand(10000, 32, 32, 1)  # 1000 张 32x32 的灰度图片
y_data = np.random.randint(0, 10, 1000)  # 数字标签 0-9

# 提取模型的特征层，移除最后的全连接层
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

# 提取特征
X_features = feature_extractor.predict(X_data)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_features, y_data, test_size=0.2, random_state=42)

# 使用轻量化分类器
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# 保存分类器为 .pkl 文件
joblib.dump(classifier, 'digit_classifier_optimized.pkl')

# 验证分类器性能
y_pred = classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# 加载保存的模型
loaded_classifier = joblib.load('digit_classifier_optimized.pkl')

# 加载新的测试图片，提取特征
# 假设 `new_image` 是新的图片，已经预处理为 (32, 32, 1)
new_image = np.random.rand(1, 32, 32, 1)  # 示例输入
new_image_features = feature_extractor.predict(new_image)

# 使用轻量化分类器预测
predicted_label = loaded_classifier.predict(new_image_features)
print(f"Predicted Label: {predicted_label[0]}")
# 转换为 TensorFlow Lite 模型
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 启用量化
tflite_model = converter.convert()

# 保存量化后的模型
with open('model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)
import tensorflow.lite as tflite
import numpy as np

# 加载 TensorFlow Lite 模型
interpreter = tflite.Interpreter(model_path='model_optimized.tflite')
interpreter.allocate_tensors()

# 获取输入/输出张量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 加载图像并预处理
preprocessed_image = np.random.rand(1, 32, 32, 1).astype('float32')  # 示例图片

# 设置输入张量并进行预测
interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
interpreter.invoke()

# 获取预测结果
prediction = interpreter.get_tensor(output_details[0]['index'])
print(f"Predicted Label: {np.argmax(prediction)}")
import cv2
from PIL import Image
import numpy as np

# 加载优化的模型
digit_model = joblib.load('digit_classifier_optimized.pkl')

# 预处理验证码图片
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # 灰度
    img = img.resize((32, 32))  # 调整大小
    img_array = np.array(img) / 255.0  # 归一化
    return img_array.reshape(1, 32, 32, 1)

# 识别单个字符
def recognize_digit(image_path):
    # 预处理图像
    image = preprocess_image(image_path)
    # 提取特征
    features = feature_extractor.predict(image)
    # 使用轻量化模型预测
    return digit_model.predict(features)[0]

# 示例：识别验证码中的字符并计算结果
image_path = ''
split_points = [0, 31, 49, 72]  # 根据图片结构调整分割点
recognized_chars = []

# 切割和识别验证码字符
captcha_image = Image.open(image_path)
for i in range(len(split_points) - 1):
    char_image = captcha_image.crop((split_points[i], 0, split_points[i + 1], captcha_image.height))
    label = recognize_digit(char_image)
    recognized_chars.append(label)

# 构建表达式并计算
expression = "".join(map(str, recognized_chars))
result = eval(expression)
print(f"The result of '{expression}' is {result}.")
