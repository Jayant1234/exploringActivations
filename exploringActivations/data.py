"""
A class to get datasets
"""
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from tensorflow.keras import backend as K
from tensorflow import keras
from scipy import misc
from scipy import ndimage
import numpy as np


class Datasets(object):
    img_rows, img_cols = 28, 28
    num_classes = 10
    def preprocess_CNN(self, x_train, x_test, y_train, y_test):

        """
        preprocesses data
        :param x_train: nparray
        :param x_test: nparray
        :param y_train: nparray
        :param y_test: nparray
        :return: tuple
        returns a tuple of preprocessed params
        """
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (self.img_rows, self.img_cols, 1)
        # convert class vectors to binary class matrices
        # y_train = keras.utils.to_categorical(y_train, self.num_classes)
        # y_test = keras.utils.to_categorical(y_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = self.normalize(x_train)
        x_test = self.normalize(x_test)
        return x_train, x_test, y_train, y_test

    def preprocess_DNN(self, x_train, x_test, y_train, y_test):
        """

        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        """
        # building the input vector from the 28x28 pixels
        x_train = x_train.reshape(x_train.shape[0], self.img_cols*self.img_rows)
        x_test = x_test.reshape(x_test.shape[0], self.img_cols*self.img_rows)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.input_shape = x_train.shape
        # normalizing the data to help with the training
        x_train = self.normalize(x_train)
        x_test = self.normalize(x_test)
        # y_train = keras.utils.to_categorical(y_train, self.num_classes)
        # y_test = keras.utils.to_categorical(y_test, self.num_classes)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def normalize(data):
        data /= 255
        return data

    @staticmethod
    def geometric_transformation(data):
        #testing
        rotate = np.random.rand(1)
        rotate_data = ndimage.rotate(data, int(rotate*360))

        return rotate_data

    @staticmethod
    def filtering(data):
        #tested : gives bad accuracy 40%
        local_mean = ndimage.uniform_filter(data)
        local_mean = data + local_mean
        return local_mean

    @staticmethod
    def denoising(data):
        #same 40
        med_denoised = ndimage.median_filter(data, 3)
        med_denoised = data + med_denoised
        return med_denoised

    @staticmethod
    def edge_detection(data):
        sy = ndimage.sobel(data, axis=1, mode='constant')
        sy = data + sy
        return sy

    def get_mnist(self, modelname):

        """
            Method to return preprocessed fashion mnist dataset
        :return:
        tuple
            A tuple of 4 nparrays is returned
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return self.preprocess_data(x_train, x_test, y_train, y_test, modelname)

    def get_fashion_mnist(self, modelname):

        """
        Method to return preprocessed fashion mnist dataset
        :return:
        tuple
           A tuple of 4 nparrays is returned
        """
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return self.preprocess_data(x_train, x_test, y_train, y_test, modelname)

    def preprocess_data(self, x_train, x_test, y_train, y_test, modelname):
        """

        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :param modelname:
        :return:
        """
        if modelname is "CNN":
            x_train, x_test, y_train, y_test = self.preprocess_CNN(x_train, x_test, y_train, y_test)
        elif modelname is "DNN":
            x_train, x_test, y_train, y_test = self.preprocess_DNN(x_train, x_test, y_train, y_test)
        return x_train, x_test, y_train, y_test
