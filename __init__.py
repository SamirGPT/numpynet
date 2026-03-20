# NumPyNet - Package principal
# Deep Learning Library en pur NumPy

from .core.layer import Layer
from .core.model import Model
from .models.sequential import Sequential
from .layers.dense import Dense
from .layers.conv2d import Conv2D, DepthwiseConv2D
from .layers.flatten import Flatten, Reshape, Permute, RepeatVector
from .layers.dropout import Dropout, SpatialDropout2D, AlphaDropout
from .layers.batch_normalization import BatchNormalization, LayerNormalization
from .layers.max_pooling2d import MaxPooling2D
from .layers.average_pooling2d import AveragePooling2D
from .activations.relu import ReLU
from .activations.sigmoid import Sigmoid
from .activations.tanh import Tanh
from .activations.softmax import Softmax
from .activations.leaky_relu import LeakyReLU
from .activations.elu import ELU
from .activations.swish import Swish
from .losses.mse import MSE, MAE, RMSE, HuberLoss, LogCosh
from .losses.binary_crossentropy import BinaryCrossentropy
from .losses.categorical_crossentropy import CategoricalCrossentropy, SparseCategoricalCrossentropy, KLDivergence, Poisson, CosineSimilarity
from .optimizers.sgd import SGD
from .optimizers.adam import Adam, Adamax, Nadam
from .optimizers.rmsprop import RMSprop
from .optimizers.adagrad import Adagrad
from .optimizers.momentum import Momentum
from .optimizers.adamw import AdamW

__version__ = '1.1.0'
__author__ = 'SamirGPT & Manus'

__all__ = [
    # Core
    'Layer',
    'Model',
    'Sequential',

    # Layers
    'Dense',
    'Conv2D',
    'DepthwiseConv2D',
    'Flatten',
    'Reshape',
    'Permute',
    'RepeatVector',
    'Dropout',
    'SpatialDropout2D',
    'AlphaDropout',
    'BatchNormalization',
    'LayerNormalization',
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
    'MAE',
    'RMSE',
    'HuberLoss',
    'LogCosh',
    'BinaryCrossentropy',
    'CategoricalCrossentropy',
    'SparseCategoricalCrossentropy',
    'KLDivergence',
    'Poisson',
    'CosineSimilarity',

    # Optimizers
    'SGD',
    'Adam',
    'Adamax',
    'Nadam',
    'RMSprop',
    'Adagrad',
    'Momentum',
    'AdamW',
]
