import re

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. 加载图像数据

# 修改的加载函数
def load_images_and_labels(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # 使用正则表达式检查文件名的第一个部分是否为数字
            match = re.match(r"(\d+)_", filename)  # 查找开头是数字的文件
            if match:
                label = int(match.group(1))
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path).convert('L')  # 转为灰度图像
                img = img.resize((32, 32))  # 调整图像大小为32x32
                img_array = np.array(img) / 255.0  # 归一化
                images.append(img_array)
                labels.append(label)
            else:
                print(f"跳过文件: {filename}, 因为它的标签部分不是数字")
    images = np.array(images).reshape(-1, 32, 32, 1)
    labels = np.array(labels)
    return images, labels

# 加载数据
directory = "G:/python-project/训练模型/segments1"  # 修改为你的数据目录
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
    tf.keras.layers.Dense(10, activation='softmax')  # 假设有10类验证码
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. 训练模型
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 5. 保存模型
model.save("num_recognition_model.h5")
print("模型已保存为 num_recognition_model.h5")
