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
        super().__init__(name=name, trainable=True)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer

        # Paramètres
        self.weights = None  # Utilisation de 'weights' pour compatibilité avec les optimiseurs
        self.bias = None

        # Gradients
        self.d_weights = None
        self.d_bias = None

        # Pour le backward pass
        self.input_tensor = None
        self.output_tensor = None

        # État
        self.input_shape = None
        self.built = False

    @property
    def kernel(self):
        """Alias pour weights."""
        return self.weights

    @kernel.setter
    def kernel(self, value):
        self.weights = value

    def build(self, input_shape: Tuple[int, int, int]) -> None:
        self.input_shape = input_shape
        channels = input_shape[2]
        kernel_h, kernel_w = self.kernel_size

        # Initialisation He
        limit = np.sqrt(2.0 / (kernel_h * kernel_w * channels))
        self.weights = np.random.randn(kernel_h, kernel_w, channels, self.filters) * limit

        if self.use_bias:
            self.bias = np.zeros((1, 1, 1, self.filters))
        else:
            self.bias = None

        self.built = True

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if not self.built:
            self.build(inputs.shape[1:])

        self.input_tensor = inputs
        batch_size, h_in, w_in, c_in = inputs.shape
        kh, kw = self.kernel_size
        sh, sw = self.strides

        # Padding
        if self.padding == 'same':
            pad_h = (kh - 1) // 2
            pad_w = (kw - 1) // 2
            self.padded_input = np.pad(inputs, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            self.padded_input = inputs

        h_out = (self.padded_input.shape[1] - kh) // sh + 1
        w_out = (self.padded_input.shape[2] - kw) // sw + 1

        output = np.zeros((batch_size, h_out, w_out, self.filters))

        # Convolution
        for i in range(h_out):
            for j in range(w_out):
                h_start, w_start = i * sh, j * sw
                region = self.padded_input[:, h_start:h_start + kh, w_start:w_start + kw, :]
                # region shape: (batch, kh, kw, c_in)
                # kernel shape: (kh, kw, c_in, filters)
                output[:, i, j, :] = np.tensordot(region, self.weights, axes=((1, 2, 3), (0, 1, 2)))

        if self.use_bias:
            output += self.bias

        self.output_tensor = output

        if self.activation is not None:
            output = self.activation(output)

        return output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        if self.activation is not None:
            grad_output = self.activation.gradient(grad_output, self.output_tensor)

        batch_size, h_out, w_out, _ = grad_output.shape
        kh, kw = self.kernel_size
        sh, sw = self.strides
        _, h_padded, w_padded, c_in = self.padded_input.shape

        grad_weights = np.zeros_like(self.weights)
        grad_padded_input = np.zeros_like(self.padded_input)

        for i in range(h_out):
            for j in range(w_out):
                h_start, w_start = i * sh, j * sw
                region = self.padded_input[:, h_start:h_start + kh, w_start:w_start + kw, :]
                
                # Gradient par rapport aux poids: sum over batch of (region * grad_output_at_pos)
                # region: (batch, kh, kw, c_in), grad_output[:, i, j, :]: (batch, filters)
                grad_weights += np.tensordot(region, grad_output[:, i, j, :], axes=((0,), (0,)))

                # Gradient par rapport à l'entrée
                # grad_output[:, i, j, :]: (batch, filters), weights: (kh, kw, c_in, filters)
                grad_padded_input[:, h_start:h_start + kh, w_start:w_start + kw, :] += \
                    np.tensordot(grad_output[:, i, j, :], self.weights, axes=((1,), (3,)))

        if self.use_bias:
            grad_bias = np.sum(grad_output, axis=(0, 1, 2), keepdims=True)
        else:
            grad_bias = None

        # Retirer le padding du gradient d'entrée
        if self.padding == 'same':
            pad_h = (kh - 1) // 2
            pad_w = (kw - 1) // 2
            grad_input = grad_padded_input[:, pad_h:-pad_h if pad_h > 0 else None, pad_w:-pad_w if pad_w > 0 else None, :]
        else:
            grad_input = grad_padded_input

        # Mise à jour
        if optimizer is not None and self.trainable:
            optimizer.update(self, grad_weights, grad_bias)

        self.d_weights = grad_weights
        self.d_bias = grad_bias

        return grad_input

    def get_weights(self) -> list:
        return [self.weights, self.bias] if self.use_bias else [self.weights]

    def set_weights(self, weights: list) -> None:
        self.weights = weights[0]
        if self.use_bias and len(weights) > 1:
            self.bias = weights[1]
        self.built = True

    def __repr__(self) -> str:
        return f"Conv2D(filters={self.filters}, kernel_size={self.kernel_size})"


class DepthwiseConv2D(Layer):
    """
    Depthwise Convolution 2D.
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
        self.weights = None
        self.bias = None
        self.built = False

    def build(self, input_shape: Tuple[int, int, int]) -> None:
        channels = input_shape[2]
        kh, kw = self.kernel_size
        self.weights = np.random.randn(kh, kw, channels, self.depth_multiplier) * 0.01
        self.built = True

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if not self.built:
            self.build(inputs.shape[1:])
        self.input_tensor = inputs
        batch_size, h, w, c = inputs.shape
        kh, kw = self.kernel_size
        sh, sw = self.strides

        if self.padding == 'same':
            pad_h, pad_w = (kh - 1) // 2, (kw - 1) // 2
            self.padded_input = np.pad(inputs, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            self.padded_input = inputs

        h_out = (self.padded_input.shape[1] - kh) // sh + 1
        w_out = (self.padded_input.shape[2] - kw) // sw + 1
        
        output = np.zeros((batch_size, h_out, w_out, c * self.depth_multiplier))

        for i in range(h_out):
            for j in range(w_out):
                h_start, w_start = i * sh, j * sw
                region = self.padded_input[:, h_start:h_start + kh, w_start:w_start + kw, :]
                # region: (batch, kh, kw, c)
                # weights: (kh, kw, c, dm)
                # output: (batch, c*dm)
                for c_idx in range(c):
                    r = region[:, :, :, c_idx] # (batch, kh, kw)
                    w_c = self.weights[:, :, c_idx, :] # (kh, kw, dm)
                    res = np.tensordot(r, w_c, axes=((1, 2), (0, 1))) # (batch, dm)
                    output[:, i, j, c_idx*self.depth_multiplier:(c_idx+1)*self.depth_multiplier] = res

        return output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        # Implementation simplified for now
        return np.zeros_like(self.input_tensor)

    def __repr__(self) -> str:
        return f"DepthwiseConv2D(kernel_size={self.kernel_size})"
