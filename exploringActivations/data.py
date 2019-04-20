"""
A class to get datasets
"""
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from tensorflow.keras import backend as K
from tensorflow import keras


class Datasets(object):
    img_rows, img_cols = 28, 28
    num_classes = 10
    def preprocess_dataset(self, x_train, x_test, y_train, y_test):

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
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        return x_train, x_test, y_train, y_test

    def get_mnist(self):

        """
            Method to return preprocessed fashion mnist dataset
        :return:
        tuple
            A tuple of 4 nparrays is returned
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test, y_train, y_test = self.preprocess_dataset(x_train, x_test, y_train, y_test)
        return x_train, x_test, y_train, y_test

    def get_fashion_mnist(self):

        """
        Method to return preprocessed fashion mnist dataset
        :return:
        tuple
           A tuple of 4 nparrays is returned
        """
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test, y_train, y_test = self.preprocess_dataset(x_train, x_test, y_train, y_test)
        return x_train, x_test, y_train, y_test
