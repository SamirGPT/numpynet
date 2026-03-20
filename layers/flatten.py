# numpynet/layers/flatten.py
"""
Flatten
=======
Aplatit l'entrée en vecteur 1D.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from ..core.layer import Layer


class Flatten(Layer):
    """
    Couche Flatten.

    Cette couche aplatit l'entrée en un vecteur 1D.
    Utilisée entre les couches convolutives et les couches denses.

    Example:
        Input:  (batch, height, width, channels) -> (32, 28, 28, 3)
        Output: (batch, height * width * channels)  -> (32, 2352)
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialise la couche Flatten.

        Args:
            name (str, optional): Nom de la couche.
        """
        super().__init__(name=name, trainable=False)
        self.input_shape = None
        self.output_shape = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant - Aplatit l'entrée.

        Args:
            inputs (np.ndarray): Données d'entrée de forme (batch, ...).
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Données aplaties de forme (batch, flatten_size).
        """
        self.input_shape = inputs.shape
        self.output_shape = (inputs.shape[0], np.prod(inputs.shape[1:]))

        self.output = inputs.reshape(self.output_shape)
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière - Remet en forme le gradient.

        Args:
            grad_output (np.ndarray): Gradient de la perte par rapport à la sortie.
            optimizer (Optimizer, optional): Non utilisé.

        Returns:
            np.ndarray: Gradient de la perte par rapport à l'entrée.
        """
        return grad_output.reshape(self.input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
        })
        return config

    def __repr__(self) -> str:
        return "Flatten()"


class Reshape(Layer):
    """
    Couche Reshape.

    Remet en forme l'entrée dans une nouvelle forme spécifiée.
    """

    def __init__(self, target_shape: Tuple[int, ...], name: Optional[str] = None):
        """
        Initialise la couche Reshape.

        Args:
            target_shape (tuple): Nouvelle forme (sans la dimension batch).
        """
        super().__init__(name=name, trainable=False)
        self.target_shape = target_shape
        self.input_shape = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Remet en forme l'entrée."""
        self.input_shape = inputs.shape
        self.output = inputs.reshape((inputs.shape[0],) + self.target_shape)
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """Remet en forme le gradient."""
        return grad_output.reshape(self.input_shape)

    def __repr__(self) -> str:
        return f"Reshape(target_shape={self.target_shape})"


class Permute(Layer):
    """
    Couche Permute.

    Permute les axes de l'entrée.
    """

    def __init__(self, perm: Tuple[int, ...], name: Optional[str] = None):
        super().__init__(name=name, trainable=False)
        self.perm = perm
        self.input_shape = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input_shape = inputs.shape
        # Ajouter batch dimension à perm
        perm_with_batch = (0,) + tuple(x + 1 for x in self.perm)
        self.output = np.transpose(inputs, perm_with_batch)
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        # Inverse permutation
        inverse_perm = np.argsort(self.perm)
        inverse_perm_with_batch = (0,) + tuple(x + 1 for x in inverse_perm)
        return np.transpose(grad_output, inverse_perm_with_batch)

    def __repr__(self) -> str:
        return f"Permute(perm={self.perm})"


class RepeatVector(Layer):
    """
    Couche RepeatVector.

    Répète le vecteur d'entrée n fois.
    """

    def __init__(self, n: int, name: Optional[str] = None):
        super().__init__(name=name, trainable=False)
        self.n = n

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.output = np.repeat(inputs, self.n, axis=1)
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        # Somme le long de l'axe répété
        return np.sum(grad_output, axis=1, keepdims=True)

    def __repr__(self) -> str:
        return f"RepeatVector(n={self.n})"
