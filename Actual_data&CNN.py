# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:36:57 2023

@author: Administrator
"""

import matplotlib.pyplot as plt
import os,PIL
import numpy as np
np.random.seed(1)
# 设置随机种子尽可能使结果可以重现
import tensorflow as tf
tf.random.set_seed(1)
from tensorflow import keras
from keras import layers,models
import pathlib

data_dir = "./Label/"                     # 路径变量
data_dir = pathlib.Path(data_dir)                  # 构造pathlib模块下的Path对象
image_count = len(list(data_dir.glob('*/*.png')))  # 使用Path对象glob方法获取所有png格式图片
print("图片总数为：",image_count)

roses = list(data_dir.glob('1/*.png'))  # 使用Path对象glob方法获取sunrise目录下所有png格式图片
PIL.Image.open(str(roses[6]))                 #显示一张图片

# 预处理
batch_size = 32
img_height = 180
img_width = 180
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,                     # 上面定义好的变量
    validation_split=0.2,         # 保留20%当做测试集
    subset="training",
    seed=123,
    image_size=(img_height, img_width),# 上面定义好的变量
    batch_size=batch_size)             # 上面定义好的变量
class_names = train_ds.class_names
print(class_names)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

#可视化
plt.figure(figsize=(20, 10))
for images, labels in train_ds.take(1):
    for i in range(20):
        ax = plt.subplot(5, 10, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

#配置训练数据集
AUTOTUNE = tf.data.AUTOTUNE                                         
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) 
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#构建卷积神经网络
num_classes = 2 
model = models.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)), # 卷积层1，卷积核3*3  
    layers.AveragePooling2D((2, 2)),               # 池化层1，2*2采样
    layers.Conv2D(32, (3, 3), activation='relu'),  # 卷积层2，卷积核3*3
    layers.AveragePooling2D((2, 2)),               # 池化层2，2*2采样
    layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层3，卷积核3*3
    layers.Dropout(0.3),  
    layers.Flatten(),                       # Flatten层，连接卷积层与全连接层
    layers.Dense(128, activation='relu'),   # 全连接层，特征进一步提取
    layers.Dense(num_classes)               # 输出层，输出预期结果
])

model.summary()  # 打印网络结构

#配置模型
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#训练模型
epochs = 10

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
model.save('D:/Desktop/test/test/model1')
#评估模型
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc,linewidth=1, label='Training Accuracy')
plt.plot(epochs_range, val_acc,linewidth=1, label='Validation Accuracy')
plt.xlabel('Epoch',fontdict={'family': 'Times New Roman', 'size': 15})
plt.ylabel("Value",fontdict={'family': 'Times New Roman', 'size': 15})
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy', fontname='Times New Roman', fontsize=15)
x_1=np.linspace(0,10,100)
y_1=np.full( 100, 0.9)
plt.plot(x_1,y_1,linewidth=1,c="red",linestyle="--")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epoch',fontdict={'family': 'Times New Roman', 'size': 15})
plt.ylabel("Value",fontdict={'family': 'Times New Roman', 'size': 15})
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', fontname='Times New Roman', fontsize=15)
x_2=np.linspace(0,10,100)
y_2=np.full(100, 0.2)
plt.plot(x_2,y_2,linewidth=1,c="red",linestyle="--")
plt.show()










