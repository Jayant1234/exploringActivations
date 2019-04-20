import sys
import argparse
from keras.utils import np_utils
from keras.models import Model
from activation_functions import CustomActivation as caf
from keras.layers import Input, Conv2D, MaxPooling2D, Upsampling2D


class ConvAutoEncoder:
    '''Creates an instance of a Convolutional Autoencoder model built using the
    Keras API
    '''
    def __init__(self, dataset_choice, activation):
        '''Initializes ConvAutoEncoder class instance.

        Arguments:
            dataset_choice (str):
                User-input choice of dataset, where, '1' refers to MNIST dataset
                and '2' refers to Fashion MNIST dataset.
            activation (str):
                User-input choice of activation function to train the model on,
                from among 'swish', 'gelu' and 'selu'.
        '''
        # If MNIST Dataset
        if dataset_choice == '1':
            from keras.datasets import mnist
            ((trainX, trainY), (testX, testY)) = mnist.load_data()
        # If Fashion MNIST Dataset
        elif dataset_choice == '2':
            from keras.datasets import fashion_mnist
            ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
            # One-hot encoding the labels
            trainY = np_utils.to_categorical(trainY, 10)
            testY = np_utils.to_categorical(testY, 10)
        self.final_shape = (trainX.shape[0], 1, 28, 28)
        trainX = trainX.reshape(self.final_shape)
        testX = testX.reshape(self.final_shape)
        # Scaling input images to the range of [0, 1]
        self.trainX = trainX.astype("float32") / 255
        self.testX = testX.astype("float32") / 255
        self.af = get_activation(activation)

    def get_activation(self, activation):
        '''Returns activation function

        Arguments:
            activation (str):
                User's choice for activation function

        Returns:
            Activation functions belonging to :class: `CustomActivation`
        '''
        af = caf()
        if activation == 'swish':
            return af.Swish()
        elif activation == 'gelu':
            return af.Gelu()
        elif activation == 'selu':
            return af.Selu()

    def compile(self):
        '''Initializes the architecture of a Convolutional Autoencoder and
        configures the model

        Returns:
            Convolutional Autoencoder model configured for training
        '''
        input_image = Input(shape=self.final_shape)
        x = Conv(16, (3, 3), activation=self.af, padding='same')(input_image)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation=self.af, padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation=self.af, padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv(8, (3, 3), activation=self.af, padding='same')(encoded)
        x = UpSampling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation=self.af, padding='same')(x)
        x = UpSampling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation=self.af, padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation=self.af, padding='same')(x)
        cae = Model(input_image, decoded)
        cae.compile(optimizer='adam', loss='binary_crossentropy')
        return cae

    def get_score(self):
        '''Fits a Convolutional AutoEncoder on the dataset and returns the
        score.

        Returns:
            score (list):
                List containing test score and test accuracy.
        '''
        model = self.compile()
        model.fit(self.trainX, self.trainY, epochs=50, batch_size=128,
            shuffle=True, validation_data=(testX, testY))
        score = model.evaluate(testX, testY, verbose=0)
        return score


def main():
    parser = argparse.ArgumentParser(
        description='Exploring Activation Functions for Image Classification')
    parser.add_argument('--dataset', dest='dataset_choice', type=str,
        choices=['1', '2'], default='1',
        help='Dataset choices: 1. MNIST, 2. MNIST Fashion')
    parser.add_argument('--af', dest='af', type=str,
        choices=['swish', 'gelu', 'selu'], default='swish')

    args = parser.parse_args()
    classifier = ConvAutoEncoder(dataset_choice=args.dataset_choice,
        activation=args.af)
    score = classifier.get_score()
    print('Test Score: ', score[0])
    print('Test Accuracy: ', score[1])


if __name__ == "__main__":
    main()
