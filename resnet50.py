# coding=utf-8
from keras.models import Model,load_model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import add, Flatten
# from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD
import numpy as np
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from pretreat import *
from img2arr import img_pretreat

seed = 7
np.random.seed(seed)


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def resnet50():
    inpt = Input(shape=(224, 224, 3))
    x = ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(216, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    model.summary()
    return model


def train():

    val_X = np.load(os.path.join(valpath,'inputs6.npy'))
    val_Y = np.load(os.path.join(valpath,'labels6.npy'))

    print('load ok')
    model = resnet50()
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)  # 优化函数，设定学习率（lr）等参数
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.load_weights(r'model\res50_216_01.h5')
    # model.fit(x=X,y=Y,batch_size=32,epochs=1000,verbose=1,shuffle=True,callbacks=callback_lists,validation_data=(val_X,val_Y))
    model.fit_generator(generate_arrays_from_file_1(trainpath, batch_size=32), epochs=100, verbose=1, workers=1, steps_per_epoch=4050,validation_data=(val_X,val_Y))
    model.save('model/res50_216class_01.h5')


def predict():

    X = img_pretreat(r'test\46901.jpg')
    X = np.reshape(X,[1,224,224,3])
    model = load_model(r'model/res50_model01_dg.h5')
    result = model.predict(X, batch_size=1, verbose=1)
    top5 = np.argsort(result)[0][-5:]
    value = np.sort(result)[0][-5:]
    with open('label_word_dict.txt', 'r', encoding='utf-8') as f:
        js = f.readlines()[0]
        print('该图片最有可能是：')
    for i in range(5):
        print(eval(js)[top5[4-i]],value[4-i])


def _test():
    model = load_model(r'model/res50_bestmodel03.h5')
    X_test = np.load(os.path.join(testpath,'inputs.npy'))
    Y_test = np.load(os.path.join(testpath,'labels.npy'))
    scores = model.evaluate(X_test, Y_test, batch_size=32, verbose=0)
    print(scores)


if __name__ == "__main__":

    train()
    # _test()
    # predict()