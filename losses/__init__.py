# numpynet/losses/__init__.py
"""
Package des fonctions de perte (Loss Functions)
=================================================
"""

from .mse import MSE
from .binary_crossentropy import BinaryCrossentropy
from .categorical_crossentropy import CategoricalCrossentropy
from .sparse_categorical_crossentropy import SparseCategoricalCrossentropy

__all__ = [
    'MSE',
    'BinaryCrossentropy',
    'CategoricalCrossentropy',
    'SparseCategoricalCrossentropy',
]
