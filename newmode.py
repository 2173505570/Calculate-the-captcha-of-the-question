import os
import re
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_images_and_labels(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            match = re.match(r"(\d+)_", filename)
            if match:
                label = int(match.group(1))
                img_path = os.path.join(directory, filename)
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize((32, 32))
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"无法加载图像文件: {img_path}, 错误: {e}")
            else:
                print(f"跳过文件: {filename}, 因为它的标签部分不是数字")

    images = np.array(images).reshape(-1, 32, 32, 1)
    labels = np.array(labels)
    return images, labels


# 加载数据
directory = "G:/python-project/训练模型/新版训练集"
images, labels = load_images_and_labels(directory)
print("图像数据维度:", images.shape)
print("标签数据维度:", labels.shape)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(14, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = create_model()
model.summary()

# 设置早停
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=64,
                    validation_split=0.2, callbacks=[early_stopping])

# 测试集评估
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("模型在测试集上的准确率:", accuracy_score(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))

# 保存模型
model.save("digit_operator_recognition_model.h5")
print("模型已保存")
