# -*- coding = utf-8 -*-
# @Time :2024/11/15 12:14
# @Author hai
# @File : operator_train.py
# @Software: PyCharm
import os
import re
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. 加载图像数据
def load_images_and_labels(directory):
    images = []
    labels = []
    label_dict = {'10': 0, '11': 1, '12': 2, '13': 3}  # 运算符到数值标签的映射

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # 使用正则表达式检查文件名的第一个部分是否为运算符名称
            match = re.match(r"(10|11|12|13)_", filename)
            if match:
                label = label_dict[match.group(1)]
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path).convert('L')  # 转为灰度图像
                img = img.resize((32, 32))  # 调整图像大小为32x32
                img_array = np.array(img) / 255.0  # 归一化
                images.append(img_array)
                labels.append(label)
            else:
                print(f"跳过文件: {filename}, 因为它的标签部分不符合预期")

    images = np.array(images).reshape(-1, 32, 32, 1)
    labels = np.array(labels)
    return images, labels

# 加载数据
directory = "G:/python-project/训练模型/segments2"  # 修改为你的数据目录
images, labels = load_images_and_labels(directory)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 3. 定义卷积神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 4类运算符
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 训练模型
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 5. 保存模型
model.save("operator_recognition_model.h5")
print("模型已保存为 operator_recognition_model.h5")
# 加载训练好的模型
# 6. 加载模型
loaded_model = tf.keras.models.load_model('operator_recognition_model.h5')

# 7. 预测新图像
def predict_image(image_path, model):
    img = Image.open(image_path).convert('L')  # 转为灰度图像
    img = img.resize((32, 32))  # 调整图像大小为32x32
    img_array = np.array(img) / 255.0  # 归一化
    img_array = img_array.reshape(1, 32, 32, 1)  # 调整形状以适应模型输入
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)  # 获取最大概率值作为置信度
    return predicted_label, confidence

# 8. 将数值标签转换为运算符符号
def label_to_operator(label):
    label_dict = {0: '+', 1: '-', 2: '*', 3: '/'}
    return label_dict[label]

# 9. 获取指定目录下所有的文件
directory = r'G:\python-project\训练模型\运算符'
files = os.listdir(directory)

# 10. 遍历所有文件
for file_name in files:
    if file_name.endswith('.png') or file_name.endswith('.jpg'):  # 检查文件是否为图像文件
        image_path = os.path.join(directory, file_name)
        predicted_label, confidence = predict_image(image_path, loaded_model)
        predicted_operator = label_to_operator(predicted_label)
        print(f"文件名: {file_name}, 预测的运算符: {predicted_operator}, 置信度: {confidence:.2f}")