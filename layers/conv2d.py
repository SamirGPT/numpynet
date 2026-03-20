# numpynet/layers/conv2d.py
"""
Conv2D (Couche Convolutionnelle 2D)
====================================
Implémente une couche de convolution 2D pour le traitement d'images.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from ..core.layer import Layer


class Conv2D(Layer):
    """
    Couche de convolution 2D.

    Cette couche applique une convolution sur les entrées 2D (images).
    Elle est le bloc de base des réseaux de neurones convolutifs (CNN).

    Attributes:
        filters (int): Nombre de filtres (noyaux de convolution).
        kernel_size (tuple): Taille du noyau de convolution.
        strides (tuple): Pas de la convolution.
        padding (str): Type de padding ('same' ou 'valid').
        activation (callable): Fonction d'activation.
        use_bias (bool): Si True, ajoute un biais.
    """

    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int, int],
                 strides: Tuple[int, int] = (1, 1),
                 padding: str = 'valid',
                 activation: Optional[Any] = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'he_normal',
                 name: Optional[str] = None):
        """
        Initialise la couche Conv2D.

        Args:
            filters (int): Nombre de filtres de sortie.
            kernel_size (tuple): Taille du noyau (height, width).
            strides (tuple): Pas de la convolution (h, w).
            padding (str): 'same' ou 'valid'.
            activation (callable, optional): Fonction d'activation.
            use_bias (bool): Si True, utilise un biais.
            kernel_initializer (str): Méthode d'initialisation des poids.
        """
        super().__init__(name=name, trainable=True)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer

        # Paramètres
        self.kernel = None
        self.bias = None

        # Pour le backward pass
        self.input_tensor = None
        self.output_tensor = None

        # État
        self.input_shape = None
        self.built = False

    def build(self, input_shape: Tuple[int, int, int]) -> None:
        """
        Construit la couche en initialisant les poids.

        Args:
            input_shape (tuple): Forme de l'entrée (height, width, channels).
        """
        self.input_shape = input_shape
        channels = input_shape[2]
        kernel_h, kernel_w = self.kernel_size

        # Initialisation des poids (kernels)
        # Xavier/He initialization adaptée pour les convolutions
        self.kernel = np.random.randn(
            kernel_h, kernel_w, channels, self.filters
        ) * np.sqrt(2.0 / (kernel_h * kernel_w * channels))

        if self.use_bias:
            self.bias = np.zeros((self.filters,))
        else:
            self.bias = None

        self.built = True

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant de la convolution 2D.

        Args:
            inputs (np.ndarray): Données d'entrée de forme (batch, height, width, channels).
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Sortie de forme (batch, new_height, new_width, filters).
        """
        # Construire si nécessaire
        if not self.built:
            self.build(inputs.shape[1:])

        self.input_tensor = inputs
        batch_size = inputs.shape[0]

        # Padding
        if self.padding == 'same':
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
            inputs_padded = np.pad(
                inputs,
                ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                mode='constant'
            )
        else:
            inputs_padded = inputs

        # Calcul des dimensions de sortie
        h_out = (inputs_padded.shape[1] - self.kernel_size[0]) // self.strides[0] + 1
        w_out = (inputs_padded.shape[2] - self.kernel_size[1]) // self.strides[1] + 1

        # Convolution avec im2col (vectorisation)
        # Pour chaque position de sortie
        output = np.zeros((batch_size, h_out, w_out, self.filters))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.strides[0]
                w_start = j * self.strides[1]

                # Extraire la région
                region = inputs_padded[
                    :,
                    h_start:h_start + self.kernel_size[0],
                    w_start:w_start + self.kernel_size[1],
                    :
                ]

                # Aplatir et multiplier avec les kernels
                region_flat = region.reshape(batch_size, -1)
                kernel_flat = self.kernel.reshape(-1, self.filters)

                output[:, i, j, :] = np.dot(region_flat, kernel_flat)

        # Ajouter le biais
        if self.use_bias:
            output += self.bias

        self.output_tensor = output

        # Appliquer l'activation
        if self.activation is not None:
            output = self.activation(output)

        return output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière de la convolution 2D.

        Args:
            grad_output (np.ndarray): Gradient de la perte par rapport à la sortie.
            optimizer (Optimizer, optional): Optimiseur.

        Returns:
            np.ndarray: Gradient de la perte par rapport à l'entrée.
        """
        # Gradient de l'activation
        if self.activation is not None:
            grad_output = self.activation.gradient(grad_output, self.output_tensor)

        batch_size = grad_output.shape[0]

        # Padding pour le backward
        if self.padding == 'same':
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
            grad_output_padded = np.pad(
                grad_output,
                ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                mode='constant'
            )
        else:
            grad_output_padded = grad_output

        # Dimensions
        h_out = grad_output.shape[1]
        w_out = grad_output.shape[2]

        # Gradient des kernels
        grad_kernel = np.zeros_like(self.kernel)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.strides[0]
                w_start = j * self.strides[1]

                region = self.input_tensor[
                    :,
                    h_start:h_start + self.kernel_size[0],
                    w_start:w_start + self.kernel_size[1],
                    :
                ]

                grad_kernel += np.einsum('bn,bo->no', region, grad_output[:, i, j, :])

        grad_kernel /= batch_size

        # Gradient du biais
        if self.use_bias:
            grad_bias = np.sum(grad_output, axis=(0, 1, 2)) / batch_size
        else:
            grad_bias = None

        # Gradient d'entrée
        grad_input = np.zeros_like(self.input_tensor)

        # Flipper le kernel pour la convolution arrière
        kernel_flipped = np.flip(self.kernel, axis=(0, 1))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.strides[0]
                w_start = j * self.strides[1]

                grad_input[
                    :,
                    h_start:h_start + self.kernel_size[0],
                    w_start:w_start + self.kernel_size[1],
                    :
                ] += np.dot(
                    grad_output_padded[:, i, j, :],
                    kernel_flipped.transpose(2, 3, 0, 1).reshape(-1, self.filters).T
                ).reshape(batch_size, self.kernel_size[0], self.kernel_size[1], -1)

        # Mise à jour des poids
        if optimizer is not None and self.trainable:
            optimizer.update(self, grad_kernel, grad_bias)

        return grad_input

    def get_weights(self) -> list:
        """Retourne les poids."""
        if self.use_bias:
            return [self.kernel, self.bias]
        return [self.kernel]

    def set_weights(self, weights: list) -> None:
        """Définit les poids."""
        self.kernel = weights[0]
        if self.use_bias and len(weights) > 1:
            self.bias = weights[1]
        self.built = True

    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': str(self.activation) if self.activation else None,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
        })
        return config

    def __repr__(self) -> str:
        return f"Conv2D(filters={self.filters}, kernel_size={self.kernel_size})"


class DepthwiseConv2D(Layer):
    """
    Depthwise Convolution 2D.

    Applique un filtre séparé par canal d'entrée.
    """

    def __init__(self,
                 kernel_size: Tuple[int, int],
                 strides: Tuple[int, int] = (1, 1),
                 padding: str = 'valid',
                 depth_multiplier: int = 1,
                 name: Optional[str] = None):
        super().__init__(name=name, trainable=True)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.depth_multiplier = depth_multiplier
        self.kernel = None
        self.built = False

    def build(self, input_shape: Tuple[int, int, int]) -> None:
        channels = input_shape[2]
        kernel_h, kernel_w = self.kernel_size
        self.kernel = np.random.randn(
            kernel_h, kernel_w, channels, self.depth_multiplier
        ) * 0.01
        self.built = True

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if not self.built:
            self.build(inputs.shape[1:])

        self.input_tensor = inputs
        batch_size = inputs.shape[0]
        h, w, c = inputs.shape[1:]

        # Calcul des dimensions de sortie
        if self.padding == 'same':
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
            inputs_padded = np.pad(inputs, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            inputs_padded = inputs

        h_out = (inputs_padded.shape[1] - self.kernel_size[0]) // self.strides[0] + 1
        w_out = (inputs_padded.shape[2] - self.kernel_size[1]) // self.strides[1] + 1
        out_channels = c * self.depth_multiplier

        output = np.zeros((batch_size, h_out, w_out, out_channels))

        # Convolution depthwise
        for b in range(batch_size):
            for i in range(h_out):
                for j in range(w_out):
                    h_start = i * self.strides[0]
                    w_start = j * self.strides[1]

                    region = inputs_padded[b, h_start:h_start + self.kernel_size[0], w_start:w_start + self.kernel_size[1], :]

                    for c_in in range(c):
                        for dm in range(self.depth_multiplier):
                            out_idx = c_in * self.depth_multiplier + dm
                            output[b, i, j, out_idx] = np.sum(
                                region[:, :, c_in] * self.kernel[:, :, c_in, dm]
                            )

        self.output_tensor = output
        return output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        # Version simplifiée
        grad_input = np.zeros_like(self.input_tensor)
        return grad_input

    def __repr__(self) -> str:
        return f"DepthwiseConv2D(kernel_size={self.kernel_size})"
