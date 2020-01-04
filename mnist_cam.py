import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2

from model import CNNModel
from cam import GradCam
from load_datasets import load_mnist

if __name__ == '__main__':
    #Load mnist
    train_num = 1000
    test_num = 100

    x_train, x_test, y_train, y_test = load_mnist(train_num, test_num)

    #Make model
    nb_filters = 64
    nb_conv = 3
    nb_pool = 2
    nb_dense = 128
    nb_classes = 10
    nb_block = 2
    input_shape = (28, 28, 1)
    cnn = CNNModel(nb_filters=nb_filters, nb_conv=nb_conv, nb_pool=nb_pool,\
        nb_dense=nb_dense, nb_classes=nb_classes, nb_block=nb_block, input_shape=input_shape)

    #fit
    epochs = 1
    batch_size = 20

    cnn.fit(x_train, y_train, x_test, y_test, epochs, batch_size)

    #cam
    model = cnn.model
    layername = 'block_1'

    plt.figure()
    for i in range(20):
        ax = plt.subplot(4, 5, i+1)
        x = x_test[i,:,:,:]
        x = x.reshape((1,28,28,1))

        prediction = model.predict(x)
        class_idx = np.argmax(prediction[0])
        print('class_idx', class_idx)

        title = 'prediction: ' + str(class_idx)
        temp = GradCam(x, model, layername)
        cam = temp.calc_cam()

        cam = cv2.resize(cam, (28, 28), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        ax.imshow(x[0,:,:,0])
        ax.imshow(cam, cmap='rainbow', alpha=0.5)
        ax.tick_params(bottom=False, left=False, right=False, top=False)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_title(title, fontsize=10)
    plt.show()


