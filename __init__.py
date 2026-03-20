# NumPyNet - Package principal
# Deep Learning Library en pur NumPy

from .core.layer import Layer
from .core.model import Model
from .models.sequential import Sequential
from .layers.dense import Dense
from .layers.conv2d import Conv2D
from .layers.flatten import Flatten
from .layers.dropout import Dropout
from .layers.batch_normalization import BatchNormalization
from .layers.max_pooling2d import MaxPooling2D
from .layers.average_pooling2d import AveragePooling2D
from .activations.relu import ReLU
from .activations.sigmoid import Sigmoid
from .activations.tanh import Tanh
from .activations.softmax import Softmax
from .activations.leaky_relu import LeakyReLU
from .activations.elu import ELU
from .activations.swish import Swish
from .losses.mse import MSE
from .losses.binary_crossentropy import BinaryCrossentropy
from .losses.categorical_crossentropy import CategoricalCrossentropy
from .losses.sparse_categorical_crossentropy import SparseCategoricalCrossentropy
from .optimizers.sgd import SGD
from .optimizers.adam import Adam
from .optimizers.rmsprop import RMSprop
from .optimizers.adagrad import Adagrad
from .optimizers.momentum import Momentum
from .optimizers.adamw import AdamW

__version__ = '1.0.0'
__author__ = 'NumPyNet Team'

__all__ = [
    # Core
    'Layer',
    'Model',
    'Sequential',

    # Layers
    'Dense',
    'Conv2D',
    'Flatten',
    'Dropout',
    'BatchNormalization',
    'MaxPooling2D',
    'AveragePooling2D',

    # Activations
    'ReLU',
    'Sigmoid',
    'Tanh',
    'Softmax',
    'LeakyReLU',
    'ELU',
    'Swish',

    # Losses
    'MSE',
    'BinaryCrossentropy',
    'CategoricalCrossentropy',
    'SparseCategoricalCrossentropy',

    # Optimizers
    'SGD',
    'Adam',
    'RMSprop',
    'Adagrad',
    'Momentum',
    'AdamW',
]
