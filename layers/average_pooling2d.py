# numpynet/layers/average_pooling2d.py
"""
AveragePooling2D
================
Couche de pooling moyen 2D.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from ..core.layer import Layer


class AveragePooling2D(Layer):
    """
    Couche Average Pooling 2D.

    Réduit la dimension spatiale en prenant la moyenne dans chaque fenêtre.

    Attributes:
        pool_size (tuple): Taille de la fenêtre de pooling.
        strides (tuple): Pas du pooling.
        padding (str): Type de padding ('same' ou 'valid').
    """

    def __init__(self,
                 pool_size: Tuple[int, int] = (2, 2),
                 strides: Optional[Tuple[int, int]] = None,
                 padding: str = 'valid',
                 name: Optional[str] = None):
        """
        Initialise la couche AveragePooling2D.

        Args:
            pool_size (tuple): Taille de la fenêtre (height, width).
            strides (tuple, optional): Pas. Si None, égal à pool_size.
            padding (str): 'same' ou 'valid'.
        """
        super().__init__(name=name, trainable=False)
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant du pooling moyen.

        Args:
            inputs (np.ndarray): Entrée de forme (batch, height, width, channels).
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Sortie de pooling.
        """
        self.input_tensor = inputs

        batch_size = inputs.shape[0]

        # Padding si nécessaire
        if self.padding == 'same':
            pad_h = (self.pool_size[0] - 1) // 2
            pad_w = (self.pool_size[1] - 1) // 2
            inputs_padded = np.pad(
                inputs,
                ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                mode='constant'
            )
        else:
            inputs_padded = inputs
            pad_h, pad_w = 0, 0

        # Dimensions de sortie
        h_out = (inputs_padded.shape[1] - self.pool_size[0]) // self.strides[0] + 1
        w_out = (inputs_padded.shape[2] - self.pool_size[1]) // self.strides[1] + 1

        # Output
        output = np.zeros((batch_size, h_out, w_out, inputs.shape[3]))

        # Pooling
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.strides[0]
                w_start = j * self.strides[1]

                region = inputs_padded[
                    :,
                    h_start:h_start + self.pool_size[0],
                    w_start:w_start + self.pool_size[1],
                    :
                ]

                # Moyenne
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

        self.output = output
        return output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière du pooling moyen.

        Le gradient est réparti uniformément sur la fenêtre.

        Args:
            grad_output (np.ndarray): Gradient de la sortie.
            optimizer (Optimizer, optional): Non utilisé.

        Returns:
            np.ndarray: Gradient de l'entrée.
        """
        batch_size = grad_output.shape[0]

        # Padding si nécessaire
        if self.padding == 'same':
            pad_h = (self.pool_size[0] - 1) // 2
            pad_w = (self.pool_size[1] - 1) // 2
        else:
            pad_h, pad_w = 0, 0

        # Gradient de la sortie padded
        grad_output_padded = np.pad(
            grad_output,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant'
        )

        # Initialiser le gradient d'entrée
        grad_input = np.zeros_like(self.input_tensor)

        # Nombre d'éléments dans la fenêtre pour la moyenne
        pool_elements = self.pool_size[0] * self.pool_size[1]

        # Propager le gradient uniformément
        for i in range(grad_output.shape[1]):
            for j in range(grad_output.shape[2]):
                h_start = i * self.strides[0]
                w_start = j * self.strides[1]

                grad_region = grad_output[:, i:i+1, j:j+1, :] / pool_elements

                grad_input[
                    :,
                    h_start:h_start + self.pool_size[0],
                    w_start:w_start + self.pool_size[1],
                    :
                ] += grad_region

        return grad_input

    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config

    def __repr__(self) -> str:
        return f"AveragePooling2D(pool_size={self.pool_size}, strides={self.strides})"


class GlobalMaxPooling2D(Layer):
    """
    Global Max Pooling 2D.

    Prend le maximum sur toute la dimension spatiale.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name, trainable=False)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input_tensor = inputs
        # Max sur les dimensions spatiales (1 et 2)
        self.output = np.max(inputs, axis=(1, 2))
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        # Créer un masque pour le maximum
        mask = (self.input_tensor == np.max(self.input_tensor, axis=(1, 2), keepdims=True))
        grad_input = mask * grad_output[:, np.newaxis, np.newaxis, :]
        return grad_input

    def __repr__(self) -> str:
        return "GlobalMaxPooling2D()"


class GlobalAveragePooling2D(Layer):
    """
    Global Average Pooling 2D.

    Prend la moyenne sur toute la dimension spatiale.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name, trainable=False)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input_tensor = inputs
        # Moyenne sur les dimensions spatiales (1 et 2)
        self.output = np.mean(inputs, axis=(1, 2))
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        # Gradient réparti uniformément
        batch_size = grad_output.shape[0]
        h, w = self.input_tensor.shape[1], self.input_tensor.shape[2]
        channels = self.input_tensor.shape[3]

        grad_input = np.zeros_like(self.input_tensor)
        for b in range(batch_size):
            for c in range(channels):
                grad_input[b, :, :, c] = grad_output[b, c] / (h * w)

        return grad_input

    def __repr__(self) -> str:
        return "GlobalAveragePooling2D()"
