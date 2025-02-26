import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


# 数据加载
def load_data(data_dir):
    images, labels = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith(('.png', '.jpg')):
            try:
                label = int(filename.split('_')[0])
                img_path = os.path.join(data_dir, filename)
                img = Image.open(img_path).convert('L').resize((28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"跳过文件 {filename}：{e}")
                continue

    if not images:
        raise ValueError("没有有效的图片文件")

    images = np.array(images)[..., np.newaxis]
    labels = np.array(labels, dtype=np.int32)
    return images, labels


data_dir = r"G:\python-project\训练模型\新版训练集"
images, labels = load_data(data_dir)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


# 构建兼容 TFLite 的轻量化模型
def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),  # TFLite 支持
        layers.MaxPooling2D((2, 2)),  # TFLite 支持
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),  # TFLite 支持
        layers.MaxPooling2D((2, 2)),  # TFLite 支持
        layers.Flatten(),
        layers.Dense(32),  # 分离 Dense 和激活
        layers.ReLU(),  # 显式使用 TFLite 支持的激活
        layers.Dropout(0.3),
        layers.Dense(14),  # 输出层
        layers.Softmax()  # 显式使用 TFLite 支持的 Softmax
    ])
    return model


model = build_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# 转换为 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("captcha_char_model.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite模型已保存到 captcha_char_model.tflite")

# 测试推理时间
interpreter = tf.lite.Interpreter(model_path="captcha_char_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

sample_input = np.expand_dims(X_test[0], axis=0).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], sample_input)

import time

start_time = time.time()
interpreter.invoke()
inference_time = (time.time() - start_time) * 1000  # 转换为ms
print(f"单次推理时间: {inference_time:.2f}ms")