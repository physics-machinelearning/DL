import keras
from keras.datasets import mnist
import numpy as np


def load_mnist(train_num, test_num):
    nb_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float')/255
    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))

    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    train_num = 1000
    test_num = 100

    randint_train = np.random.randint(0, x_train.shape[0], train_num)
    randint_test = np.random.randint(0, x_test.shape[0], test_num)

    x_train = x_train[randint_train,:,:,:]
    x_test = x_test[randint_test,:,:,:]
    y_train = y_train[randint_train,:]
    y_test = y_test[randint_test,:]

    return x_train, x_test, y_train, y_test