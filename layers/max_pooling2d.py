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
    """

    def __init__(self,
                 pool_size: Tuple[int, int] = (2, 2),
                 strides: Optional[Tuple[int, int]] = None,
                 padding: str = 'valid',
                 name: Optional[str] = None):
        super().__init__(name=name, trainable=False)
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding

        # Pour le backward pass
        self.input_tensor = None
        self.mask = None 

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input_tensor = inputs
        batch_size, h_in, w_in, channels = inputs.shape
        kh, kw = self.pool_size
        sh, sw = self.strides

        # Padding
        if self.padding == 'same':
            pad_h, pad_w = (kh - 1) // 2, (kw - 1) // 2
            self.padded_input = np.pad(inputs, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            self.padded_input = inputs
            pad_h, pad_w = 0, 0

        h_out = (self.padded_input.shape[1] - kh) // sh + 1
        w_out = (self.padded_input.shape[2] - kw) // sw + 1

        output = np.zeros((batch_size, h_out, w_out, channels))
        self.mask = np.zeros_like(self.padded_input)

        for i in range(h_out):
            for j in range(w_out):
                h_start, w_start = i * sh, j * sw
                h_end, w_end = h_start + kh, w_start + kw
                region = self.padded_input[:, h_start:h_end, w_start:w_end, :]
                
                max_vals = np.max(region, axis=(1, 2), keepdims=True)
                output[:, i, j, :] = max_vals.squeeze(axis=(1, 2))
                
                # Mask for backward
                self.mask[:, h_start:h_end, w_start:w_end, :] += (region == max_vals)

        return output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        kh, kw = self.pool_size
        sh, sw = self.strides
        
        grad_padded_input = np.zeros_like(self.padded_input)
        h_out, w_out = grad_output.shape[1], grad_output.shape[2]

        for i in range(h_out):
            for j in range(w_out):
                h_start, w_start = i * sh, j * sw
                h_end, w_end = h_start + kh, w_start + kw
                
                # Propagate gradient to the max positions
                # grad_output[:, i, j, :] shape: (batch, channels)
                # mask shape: (batch, kh, kw, channels)
                grad_padded_input[:, h_start:h_end, w_start:w_end, :] += \
                    self.mask[:, h_start:h_end, w_start:w_end, :] * grad_output[:, i:i+1, j:j+1, :]

        if self.padding == 'same':
            pad_h, pad_w = (kh - 1) // 2, (kw - 1) // 2
            grad_input = grad_padded_input[:, pad_h:-pad_h if pad_h > 0 else None, pad_w:-pad_w if pad_w > 0 else None, :]
        else:
            grad_input = grad_padded_input

        return grad_input

    def __repr__(self) -> str:
        return f"MaxPooling2D(pool_size={self.pool_size})"
