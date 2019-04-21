
from tensorflow.distributions import Normal
from keras.activations import sigmoid, selu
from keras import backend as K


class CustomActivation:
    '''Implements custom activation functions
    References:
    1. https://arxiv.org/pdf/1710.05941.pdf
    2. https://arxiv.org/pdf/1606.08415.pdf
    '''
    def Swish(self, x):
        '''Implementation for Swish Activation Function
        
        Arguments:
            x (tensor):
                Input tensor
        Returns:
            Tensor, output of 'Swish' activation function
        '''
        return K.sigmoid(x) * x

    def Gelu(self, x):
        '''Implementation for GELU Activation Function

        Arguments:
            x (tensor):
                Input tensor
        Returns:
            Tensor, output of 'GELU' activation function
        '''
        normal = Normal(loc=0.,
                        scale=1.)
        return x * normal.cdf(x)

    def Selu(self, x):
        '''Composes Keras' implementation for SELU Activation Function

        Arguments:
            x (tensor):
                Input tensor

        Returns:
            Tensor, output of 'SELU' activatin function 
        '''
        return selu(x)
