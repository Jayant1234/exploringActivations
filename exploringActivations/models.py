"""
Class to define deep learning models
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras import optimizers
from activation_functions import CustomActivation
from data import Datasets
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model


class DeepModels:

    def __init__(self, activation_name="swish"):
        act = CustomActivation()
        if activation_name == "gelu":
            self.act_method = act.Gelu
        elif activation_name == "selu":
            self.act_method = act.Selu
        elif activation_name == "swish":
            self.act_method = act.Swish
        elif activation_name == 'relu':
            self.act_method = 'relu'
        elif activation_name == 'sigmoid':
            self.act_method = 'sigmoid'

    def define_model(self):
        pass

    def compile_model(self, model):
        model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model

    def fit_model(self, model, x_train, x_test, y_train, y_test):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        history = model.fit(x_train, y_train,
                  batch_size=128,
                  epochs=500,
                  verbose=1,
                  validation_data=(x_test, y_test), callbacks=[es])
        return model, history

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
        model.add(Dense(10,  activation='softmax'))
        return model

    def run_model(self, input_shape, x_train, x_test, y_train, y_test):
        model = self.define_model(input_shape)
        model = self.compile_model(model)
        model, history = self.fit_model(model, x_train, x_test, y_train, y_test)
        score = self.evaluate_model(model, x_test, y_test)
        return score, history


class DNN(DeepModels):

    def define_model(self, input_shape, hidden_layers=1):
        instances, x_dim = input_shape
        model = Sequential()
        hidden_max = 500
        for i in range(hidden_layers):
            hidden_units = hidden_max/(i+1)
            if hidden_units < 100:
                hidden_units = 100
            model.add(Dense(units=hidden_units, input_shape=(x_dim,), activation=self.act_method))
        model.add(Dense(units=10,  activation='softmax'))
        return model

    def run_model(self, input_shape, x_train, x_test, y_train, y_test, hidden_layers=1):
        model = self.define_model(input_shape, hidden_layers)
        model = self.compile_model(model)
        model, history = self.fit_model(model, x_train, x_test, y_train, y_test)
        score = self.evaluate_model(model, x_test, y_test)
        return score, history


class CAE(DeepModels):

    def define_model(self, input_shape):

        input_image = Input(shape=input_shape)
        x = Conv2D(16, (3, 3), activation=self.act_method, padding='same')(input_image)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation=self.act_method, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation=self.act_method, padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(8, (3, 3), activation=self.act_method, padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation=self.act_method, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation=self.act_method, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation=self.act_method, padding='same')(x)
        flat = Flatten()(decoded)
        x = Dense(10, activation='softmax')(flat)
        cae = Model(input_image, x)
        return cae

    def run_model(self, input_shape, x_train, x_test, y_train, y_test):

        model = self.define_model(input_shape)
        model = self.compile_model(model)
        model, history = self.fit_model(model, x_train, x_test, y_train, y_test)
        score = self.evaluate_model(model, x_test, y_test)
        return score, history



