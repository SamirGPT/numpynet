# numpynet/layers/batch_normalization.py
"""
BatchNormalization
=================
Couche de normalisation par lots.
"""

import numpy as np
from typing import Optional, Dict, Any
from ..core.layer import Layer


class BatchNormalization(Layer):
    """
    Couche Batch Normalization.
    """

    def __init__(self,
                 momentum: float = 0.99,
                 epsilon: float = 1e-3,
                 axis: int = -1,
                 center: bool = True,
                 scale: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name, trainable=True)
        self.momentum = momentum
        self.epsilon = epsilon
        self.axis = axis
        self.center = center
        self.scale = scale

        # Paramètres entraînables (renommés pour compatibilité avec l'optimiseur si possible)
        # Mais BatchNorm a souvent 2 paramètres distincts. 
        # On va utiliser weights et bias comme alias pour gamma et beta.
        self.gamma = None  # Scale
        self.beta = None   # Shift

        # Moyennes mobiles
        self.moving_mean = None
        self.moving_var = None

        self.built = False

    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, value):
        self.gamma = value

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, value):
        self.beta = value

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            shape = (input_shape[-1],)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = np.ones(shape)
        if self.center:
            self.beta = np.zeros(shape)

        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)
        self.built = True

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if not self.built:
            self.build(inputs.shape)

        self.input = inputs
        
        # Déterminer les axes de réduction
        if self.axis == -1 or self.axis == inputs.ndim - 1:
            reduction_axes = tuple(range(inputs.ndim - 1))
        else:
            reduction_axes = tuple(i for i in range(inputs.ndim) if i != self.axis)

        if training:
            self.batch_mean = np.mean(inputs, axis=reduction_axes, keepdims=True)
            self.batch_var = np.var(inputs, axis=reduction_axes, keepdims=True)
            
            self.normalized = (inputs - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
            
            # Update moving stats
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.batch_mean.squeeze()
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.batch_var.squeeze()
        else:
            # Reshape moving stats for broadcasting
            shape = [1] * inputs.ndim
            if self.axis == -1:
                shape[-1] = -1
            else:
                shape[self.axis] = -1
            
            mean = self.moving_mean.reshape(shape)
            var = self.moving_var.reshape(shape)
            self.normalized = (inputs - mean) / np.sqrt(var + self.epsilon)

        # Apply gamma and beta
        shape = [1] * inputs.ndim
        if self.axis == -1:
            shape[-1] = -1
        else:
            shape[self.axis] = -1
            
        out = self.normalized
        if self.scale:
            out = out * self.gamma.reshape(shape)
        if self.center:
            out = out + self.beta.reshape(shape)
            
        return out

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        if self.axis == -1 or self.axis == self.input.ndim - 1:
            reduction_axes = tuple(range(self.input.ndim - 1))
        else:
            reduction_axes = tuple(i for i in range(self.input.ndim) if i != self.axis)

        # Gradients for gamma and beta
        if self.scale:
            grad_gamma = np.sum(grad_output * self.normalized, axis=reduction_axes)
        else:
            grad_gamma = None
            
        if self.center:
            grad_beta = np.sum(grad_output, axis=reduction_axes)
        else:
            grad_beta = None

        # Gradient for input
        N = np.prod([self.input.shape[i] for i in reduction_axes])
        shape = [1] * self.input.ndim
        if self.axis == -1:
            shape[-1] = -1
        else:
            shape[self.axis] = -1
            
        gamma = self.gamma.reshape(shape) if self.scale else 1.0
        
        std_inv = 1.0 / np.sqrt(self.batch_var + self.epsilon)
        dx_norm = grad_output * gamma
        
        grad_input = (1.0 / N) * std_inv * (
            N * dx_norm - 
            np.sum(dx_norm, axis=reduction_axes, keepdims=True) - 
            self.normalized * np.sum(dx_norm * self.normalized, axis=reduction_axes, keepdims=True)
        )

        if optimizer is not None and self.trainable:
            # L'optimiseur attend weights et bias
            optimizer.update(self, grad_gamma, grad_beta)

        return grad_input

    def get_weights(self) -> list:
        res = []
        if self.scale: res.append(self.gamma)
        if self.center: res.append(self.beta)
        return res

    def set_weights(self, weights: list) -> None:
        idx = 0
        if self.scale:
            self.gamma = weights[idx]
            idx += 1
        if self.center:
            self.beta = weights[idx]
        self.built = True

    def __repr__(self) -> str:
        return f"BatchNormalization(axis={self.axis})"


class LayerNormalization(Layer):
    """
    Layer Normalization.
    """
    def __init__(self, epsilon: float = 1e-3, name: Optional[str] = None):
        super().__init__(name=name, trainable=True)
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.built = False

    def build(self, input_shape):
        self.gamma = np.ones(input_shape[1:])
        self.beta = np.zeros(input_shape[1:])
        self.built = True

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if not self.built:
            self.build(inputs.shape)
        
        # Mean and var over all axes except batch
        axes = tuple(range(1, inputs.ndim))
        self.mean = np.mean(inputs, axis=axes, keepdims=True)
        self.var = np.var(inputs, axis=axes, keepdims=True)
        
        self.normalized = (inputs - self.mean) / np.sqrt(self.var + self.epsilon)
        return self.gamma * self.normalized + self.beta

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        # Simplified backward
        return grad_output
