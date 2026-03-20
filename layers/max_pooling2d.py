# numpynet/layers/max_pooling2d.py
"""
MaxPooling2D
===========
Couche de pooling maximal 2D.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from ..core.layer import Layer


class MaxPooling2D(Layer):
    """
    Couche Max Pooling 2D.

    Réduit la dimension spatiale en prenant la valeur maximale dans chaque fenêtre.

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
        Initialise la couche MaxPooling2D.

        Args:
            pool_size (tuple): Taille de la fenêtre (height, width).
            strides (tuple, optional): Pas. Si None, égal à pool_size.
            padding (str): 'same' ou 'valid'.
        """
        super().__init__(name=name, trainable=False)
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding

        # Pour le backward pass
        self.input_tensor = None
        self.mask = None  # Pour stocker les positions des max

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant du pooling maximal.

        Args:
            inputs (np.ndarray): Entrée de forme (batch, height, width, channels).
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Sortie de pooling.
        """
        self.input_tensor = inputs

        batch_size = inputs.shape[0]
        h_in, w_in, channels = inputs.shape[1:]

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
        output = np.zeros((batch_size, h_out, w_out, channels))
        self.mask = np.zeros_like(inputs_padded)

        # Pooling
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.strides[0]
                w_start = j * self.strides[1]

                h_end = h_start + self.pool_size[0]
                w_end = w_start + self.pool_size[1]

                region = inputs_padded[:, h_start:h_end, w_start:w_end, :]

                # Trouver le maximum et sa position
                max_vals = np.max(region, axis=(1, 2))
                output[:, i, j, :] = max_vals

                # Créer le mask pour le backward
                max_mask = (region == max_vals[:, np.newaxis, np.newaxis, :])
                self.mask[:, h_start:h_end, w_start:w_end, :] += max_mask

        self.output = output
        return output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière du pooling maximal.

        Args:
            grad_output (np.ndarray): Gradient de la sortie.
            optimizer (Optimizer, optional): Non utilisé.

        Returns:
            np.ndarray: Gradient de l'entrée.
        """
        # Dimensions
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

        # Propager le gradient aux positions du max
        for i in range(grad_output.shape[1]):
            for j in range(grad_output.shape[2]):
                h_start = i * self.strides[0]
                w_start = j * self.strides[1]
                h_end = h_start + self.pool_size[0]
                w_end = w_start + self.pool_size[1]

                mask_region = self.mask[:, h_start:h_end, w_start:w_end, :]
                grad_region = grad_output[:, i:i+1, j:j+1, :]

                grad_input[:, h_start:h_end, w_start:w_end, :] += mask_region * grad_region

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
        return f"MaxPooling2D(pool_size={self.pool_size}, strides={self.strides})"
