"""
Class to define deep learning models
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from activation_functions import CustomActivation
from data import Datasets
from tensorflow import keras

class DeepModels:

    def __init__(self, activation_name="swish"):
        act = CustomActivation()
        if activation_name == "gelu":
            self.act_method = act.Gelu
        elif activation_name == "selu":
            self.act_method = act.Selu
        else:
            self.act_method = act.Swish

    def define_model(self):
        pass

    def compile_model(self, model):
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model

    def fit_model(self, model, x_train, x_test, y_train, y_test):
        model.fit(x_train, y_train,
                  batch_size=128,
                  epochs=12,
                  verbose=1,
                  validation_data=(x_test, y_test))
        return model

    def save_model(self, model):
        pass

    def evaluate_model(self, model, x_test, y_test):
        score = model.evaluate(x_test, y_test, verbose=0)
        return score


class CNN(DeepModels):

    def define_model(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation=self.act_method,
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation=self.act_method))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation=self.act_method))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation=self.act_method))
        return model

    def run_model(self, input_shape, x_train, x_test, y_train, y_test):
        model = self.define_model(input_shape)
        model = self.compile_model(model)
        model = self.fit_model(model, x_train, x_test, y_train, y_test)
        score = self.evaluate_model(model, x_test, y_test)
        return score


class DNN(DeepModels):

    def define_model(self, input_shape):
        instances, x_dim = input_shape
        model = Sequential()
        model.add(Dense(500, input_dim=x_dim, activation=self.act_method,
                        kernel_initializer=keras.initializers.RandomNormal(stddev=1), bias_initializer='zeros'))
        model.add(Dense(400, activation=self.act_method, kernel_initializer=keras.initializers.RandomNormal(stddev=1),
                        bias_initializer='zeros'))
        model.add(Dense(200, activation=self.act_method, kernel_initializer=keras.initializers.RandomNormal(stddev=1),
                        bias_initializer='zeros'))
        model.add(Dense(10, activation=self.act_method, kernel_initializer=keras.initializers.RandomNormal(stddev=1),
                        bias_initializer='zeros'))
        return model

    def run_model(self, input_shape, x_train, x_test, y_train, y_test):
        model = self.define_model(input_shape)
        model = self.compile_model(model)
        model = self.fit_model(model, x_train, x_test, y_train, y_test)
        score = self.evaluate_model(model, x_test, y_test)
        return score





