from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

class CNNModel:
    def __init__(self, nb_filters, nb_conv, nb_pool, nb_dense, nb_classes, nb_block, input_shape):
        self.nb_filters = nb_filters
        self.nb_conv = nb_conv
        self.nb_pool = nb_pool
        self.nb_dense = nb_dense
        self.nb_classes = nb_classes
        self.nb_block = nb_block
        self.input_shape = input_shape
        self.model = self.make_model()
        
    def block(self, inp, i):
        name = 'block_' + str(i)
        x = Conv2D(self.nb_filters, self.nb_conv)(inp)
        x = MaxPooling2D(self.nb_pool, name=name)(x)
        x = Activation('relu')(x)
        out = Dropout(0.2)(x)
        return out

    def make_model(self):
        inp = Input(shape=self.input_shape)
        x = inp

        for i in range(self.nb_block):
            x = self.block(x,i)

        x = Flatten()(x)
        x = Dense(self.nb_dense)(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(self.nb_classes)(x)
        out = Activation('softmax')(x)

        cnn = Model(inp, out)
        return cnn

    def fit(self, x_train, y_train, x_test, y_test, epochs, batch_size):
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        self.model.fit(x_train, y_train, nb_epoch=epochs, batch_size=batch_size, shuffle=True,
                        validation_data=(x_test, y_test))

    def predict(self, x_test):
        y_test_predicted = self.model.predict(x_test)
        return y_test_predicted

class GANModel:
    def __init__(self, input_dim, x_shape, g_block_num, d_block_num, d_dense_num):
        self.input_dim = input_dim
        self.x_shape = x_shape
        self.g_block_num = g_block_num
        self.d_blocknum = d_block_num
        self.d_densenum = d_dense_num

        assert self.x_shape[0]%(2**(self.g_block_num+1))==0, 'Dimension error'

        self.d = self.discriminator()
        self.g = self.generator()
        self.gan = self.gan_model()

    def generator(self):
        def block(inp):
            x = Conv2DTranspose(64,(2,2), strides=(2,2), padding='same')(inp)
            x = Conv2D(64,(3,3), padding='same')(x)
            x = BatchNormalization()(x)
            out = Activation('relu')(x)
            return out

        first_dim = self.x_shape[0]//(2**(self.g_block_num+1))

        inputs = Input((self.input_dim,))
        fc1 = Dense(128*first_dim*first_dim)(inputs)
        fc1 = BatchNormalization()(fc1)
        fc1 = LeakyReLU(0.2)(fc1)
        x = Reshape((first_dim,first_dim,128), input_shape=(128*first_dim*first_dim,))(fc1)
        
        for i in range(self.g_block_num):
            x = block(x)

        x = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(x)
        x = Conv2D(1, (3,3), padding='same')(x)
        out = Activation('tanh')(x)
    
        model = Model(inputs=[inputs], outputs=[out])
        return model

    def discriminator(self):
        def block(inp):
            conv = Conv2D(64, (5,5), padding='same')(inp)
            conv = LeakyReLU(0.2)(conv)
            pool = MaxPooling2D(pool_size=(2,2))(conv)
            return pool

        inp = Input((self.x_shape[0],self.x_shape[1],1))
        x = inp

        for i in range(self.d_blocknum):
            x = block(x)

        fc = Flatten()(x)

        fc_shape = K.int_shape(fc)[1]
 
        if self.d_densenum == 1:
            pass
        else:
            for i in range(self.d_densenum-1):
                num = fc_shape//(self.d_densenum)*(i+1)
                fc = Dense(num)(fc)
        
        fc = Dense(1)(fc)
        out = Activation('sigmoid')(fc)
    
        model = Model(inputs = [inp], outputs = [out])
        return model

    def gan_model(self):
        self.d.trainable = False
        ginp = Input((self.input_dim, ))
        gout = self.g(ginp)
        dout = self.d(gout)
        model = Model(inputs=ginp, outputs=dout) 
        return model

    def train(self, epochs, batch_size, x_train):
        d_optim = RMSprop(lr=0.0004)
        gan_optim = RMSprop(lr=0.0002)
        self.gan.compile(loss='binary_crossentropy', optimizer=gan_optim)
        self.d.trainable = True
        self.d.compile(loss='mse', optimizer=d_optim)
        
        n_iter = int(x_train.shape[0]/batch_size)

        for epoch in range(epochs):
            print(epoch)
            for index in range(n_iter):
                #Create imgs from noise 
                noise = np.random.uniform(0,1,size=(batch_size, self.input_dim))
                imgs_batch = x_train[index*batch_size:(1+index)*batch_size]
                generated_imgs = self.g.predict(noise)

                #Make data for training discriminator
                x = np.concatenate((imgs_batch, generated_imgs))
                y = np.array([1]*batch_size+[0]*batch_size)

                #Train discriminator
                self.d.train_on_batch(x, y)

                #Train generator
                self.d.trainable = False
                self.gan.train_on_batch(noise, np.array([1]*batch_size))
                self.d.trainable = True

            plt.figure()
            for i in range(4):
                plt.subplot(2,2,i+1)
                plt.imshow(generated_imgs[i,:,:,:].reshape((28,28)))
            plt.subplot()

            self.g.save_weights('./weights/generator.h5', True)
            self.d.save_weights('./weights/discriminator.h5', True)

    def load_model(self):
        d = self.discriminator()
        g = self.generator()
        d.load_weights('./weights/discriminator.h5')
        g.load_weights('./weights/generator.h5')
        return g, d

if __name__ == '__main__':
    input_dim = 10
    x_shape = (28, 28)
    g_block_num = 1
    d_block_num = 2
    d_dense_num = 2
    gan = GANModel(input_dim=input_dim, x_shape=x_shape, g_block_num=g_block_num, 
    d_block_num=d_block_num, d_dense_num=d_dense_num)
    g = gan.generator()
    d = gan.discriminator()

    