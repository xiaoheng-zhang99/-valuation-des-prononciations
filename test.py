import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras import regularizers
import os
import keras

import keras.backend as K

data = "/Users/zhangxiaoheng/Desktop/new/data.csv"

df = pd.read_csv(data, header=0, index_col=0)
df1 = df.drop(["y"], axis=1)
lbls = df["y"].values - 1

wave = np.zeros((11500, 178))

z = 0
for index, row in df1.iterrows():
    wave[z, :] = row
    z += 1

mean = wave.mean(axis=0)
wave -= mean
std = wave.std(axis=0)
wave /= std


# print(wave)  # 为什么要进行标准化


def one_hot(y):
    lbl = np.zeros(5)
    lbl[y] = 1
    return lbl


target = []
for value in lbls:
    target.append(one_hot(value))
target = np.array(target)
# .array，列表中的元素可以是任何对象，因此列表中保存的是对象的指针，这样一来，为了保存一个简单的列表[1,2,3]。就需要三个指针和三个整数对象。对于数值运算来说，这种结构显然不够高效 expand_dims的含义
wave = np.expand_dims(wave, axis=-1)  # 在最后增加一个维度
# print(target.shape)

# sequential？通过将网络层实例的列表传递给 Sequential 的构造器，来创建一个 Sequential 模型
from keras.models import Sequential
from keras import layers

model = Sequential()
model.add(layers.Conv1D(64, 15, strides=2, input_shape=(178, 1), use_bias=False))
model.add(layers.ReLU())
model.add(layers.Conv1D(64, 3))
model.add(layers.Conv1D(64, 3, strides=2))
model.add(layers.ReLU())
model.add(layers.Conv1D(64, 3))
model.add(layers.Conv1D(64, 3, strides=2))  # [None, 54, 64]
model.add(layers.BatchNormalization())
model.add(layers.LSTM(64, dropout=0.5, return_sequences=True))
model.add(layers.LSTM(64, dropout=0.5, return_sequences=True))
model.add(layers.LSTM(32))
model.add(layers.Dense(5, activation="softmax"))
# model.summary()
# strides dropout 含义
# batch normalization and .dense
# 1D 特征提取 LSTM特征建模
# activation 复习各种激活函数


save_path = '/Users/zhangxiaoheng/Desktop/new/keras_model.h5'

if os.path.isfile(save_path):
    model.load_weights(save_path)
    print('reloaded.')

adam = tf.keras.optimizers.Adam()
# adam 是啥 优化器？ loss和metrics
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["acc"])


# 计算学习率
def lr_scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的0.5
    # 为什么要减小学习率？model.optimizer.lr
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)


lrate = keras.callbacks.LearningRateScheduler(lr_scheduler)

# keras 本质用来干啥的 LearningRateScheduler？callback? batch size如何确定 history.history.keys()

history = model.fit(wave, target, epochs=400,
                    batch_size=128, validation_split=0.2,
                    verbose=1, callbacks=[lrate])

model.save_weights(save_path)

# demo
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# keras model.h5? cnn+lstm?
