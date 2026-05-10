import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import cv2

# ======================
# 1. 数据集路径（Kaggle）
# ======================
base_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# ======================
# 2. 数据增强 & 加载
# ======================
img_size = 150
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # 直接从训练集划8:2！！！（作业强制要求）
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 训练集
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

# 验证集
val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

# 测试集
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=1,
    class_mode="binary",
    shuffle=False
)

# ======================
# 3. 搭建CNN模型
# ======================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# 编译
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# 4. 训练模型
# ======================
epochs = 15
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# ======================
# 5. 绘制训练曲线
# ======================
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy Curve")
plt.legend()
plt.savefig("/kaggle/working/training_curve.png")
plt.show()
plt.close()

# ======================
# 6. 测试集评估
# ======================
test_generator.reset()
preds = model.predict(test_generator, verbose=1)
y_pred = (preds > 0.5).astype(int).flatten()
y_true = test_generator.classes

# 指标
print(classification_report(y_true, y_pred, target_names=["Normal","Pneumonia"]))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Normal","Pneumonia"],
            yticklabels=["Normal","Pneumonia"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig("/kaggle/working/confusion_matrix.png")
plt.show()
plt.close()
