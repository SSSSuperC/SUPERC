import numpy as np
import tensorflow as tf
import os
import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
import math
import random
# strategy = tf.distribute.MirroredStrategy()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

#变量初始化
batch_size = 128
nb_classes = 10

#数据预处理阶段
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



def drawDigit(position, image, title):
    plt.subplot(*position)
    plt.imshow(image.reshape(-1, 28), cmap='gray_r')
    plt.axis('off')
    plt.title(title)


def batchDraw(batch_size):
    selected_index = random.sample(range(len(y_train)), k=batch_size)
    images, labels = x_train[selected_index], y_train[selected_index]
    image_number = images.shape[0]
    row_number = math.ceil(image_number ** 0.5)
    column_number = row_number
    plt.figure(figsize=(row_number, column_number))
    for i in range(row_number):
        for j in range(column_number):
            index = i * column_number + j
            if index < image_number:
                position = (row_number, column_number, index + 1)
                image = images[index]
                actual = np.argmax(labels[index])
                title = 'actual:%d' % (actual)
                drawDigit(position, image, title)


batchDraw(100)
plt.show()

# with strategy.scope():
#模型构建
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])


#模型可视化
print(model.summary())
#模型编译
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['acc'])

# 转换
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#创建一个实例history
history = LossHistory()

#训练模型
model.fit(x_train, y_train, validation_split=0.2,epochs=5,verbose=1,callbacks=[history])

#模型评估

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

#绘制acc-loss曲线
history.loss_plot('epoch')

#模型测试
def drawDigit3(position, image, title):
    plt.subplot(*position)
    plt.imshow(image.reshape(-1, 28), cmap='gray_r')
    plt.axis('off')
    plt.title(title)


def batchDraw3(batch_size, test_X, test_y):
    selected_index = random.sample(range(len(test_y)), k=100)
    images = test_X[selected_index]
    labels = test_y[selected_index]
    predict_labels = model.predict(images)
    image_number = images.shape[0]
    row_number = math.ceil(image_number ** 0.5)
    column_number = row_number
    plt.figure(figsize=(row_number + 8, column_number + 8))
    for i in range(row_number):
        for j in range(column_number):
            index = i * column_number + j
            if index < image_number:
                position = (row_number, column_number, index + 1)
                image = images[index]
                predict = np.argmax(predict_labels[index])
                title = 'predict:%d' % (predict)
                drawDigit3(position, image, title)


batchDraw3(100, x_test, y_test)
plt.show()
