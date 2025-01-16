import os
import numpy as np
from PIL import Image
import tensorflow as tf

# 加载训练好的模型
model = tf.keras.models.load_model("num_recognition_model.h5")

# 预测函数
def predict_image(model, image_path):
    # 加载图像并预处理
    img = Image.open(image_path).convert('L')  # 转为灰度图像
    img = img.resize((32, 32))  # 调整图像大小
    img_array = np.array(img) / 255.0  # 归一化
    img_array = img_array.reshape(1, 32, 32, 1)  # 调整形状以适配模型输入

    # 预测标签
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)
    return predicted_label, confidence

# 获取指定目录下所有的文件
# directory = r'G:\python-project\训练模型\images'
# files = os.listdir(directory)
image_path = r'G:\python-project\训练模型\output\region_0.png'
label, confidence = predict_image(model, image_path)
print(f" 预测的标签: {label}, 置信度: {confidence:.2f}")
# # 遍历所有文件
# for file_name in files:
#     if file_name.endswith('.png') or file_name.endswith('.jpg'):  # 检查文件是否为图像文件
#         image_path = os.path.join(directory, file_name)
#         label, confidence = predict_image(model, image_path)
#         print(f"文件名: {file_name}, 预测的标签: {label}, 置信度: {confidence:.2f}")