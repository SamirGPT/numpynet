# numpynet/activations/swish.py
"""
Swish
=====
Fonction d'activation découverte par recherche automatique.
Formule: f(x) = x * sigmoid(x)
Dérivée: f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
"""

import numpy as np
from typing import Optional, Any
from ..core.layer import Layer


class Swish(Layer):
    """
    Activation Swish.

    Formule: f(x) = x * σ(x) = x / (1 + e^(-x))

    Caractéristiques:
        - Non monotonique (petit "bump" près de 0)
        - Plus performant que ReLU sur certains benchmarks
        - Plus coûteux compututionnellement
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialise l'activation Swish.

        Args:
            name (str, optional): Nom de la couche.
        """
        super().__init__(name=name, trainable=False)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant de Swish.

        Args:
            inputs (np.ndarray): Tension d'entrée.
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Sortie de Swish.
        """
        self.input = inputs
        sigmoid_inputs = 1 / (1 + np.exp(-inputs))
        self.output = inputs * sigmoid_inputs
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière de Swish.

        Args:
            grad_output (np.ndarray): Gradient de la perte par rapport à la sortie.
            optimizer (Optimizer, optional): Non utilisé.

        Returns:
            np.ndarray: Gradient de la perte par rapport à l'entrée.
        """
        sigmoid_inputs = 1 / (1 + np.exp(-self.input))
        # Dérivée: σ(x) + x * σ(x) * (1 - σ(x))
        swish_derivative = sigmoid_inputs + self.input * sigmoid_inputs * (1 - sigmoid_inputs)
        grad_input = grad_output * swish_derivative
        return grad_input

    def __repr__(self) -> str:
        return "Swish()"


class Mish(Layer):
    """
    Activation Mish.

    Formule: f(x) = x * tanh(softplus(x))
    où softplus(x) = ln(1 + e^x)

    Caractéristiques:
        - Similaire à Swish mais utilise tanh au lieu de sigmoid
        - Plus lisse que Swish
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name, trainable=False)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = inputs
        softplus = np.log(1 + np.exp(np.clip(inputs, -500, 500)))
        tanh_softplus = np.tanh(softplus)
        self.output = inputs * tanh_softplus
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        softplus = np.log(1 + np.exp(np.clip(self.input, -500, 500)))
        sigmoid_sp = 1 / (1 + np.exp(-softplus))
        tanh_sp = np.tanh(softplus)

        # Dérivée: f'(x) = tanh(softplus) + x * sigmoid(softplus) * (1 - tanh(softplus)²)
        mish_derivative = tanh_sp + self.input * sigmoid_sp * (1 - tanh_sp * tanh_sp)
        grad_input = grad_output * mish_derivative
        return grad_input

    def __repr__(self) -> str:
        return "Mish()"


class GELU(Layer):
    """
    Activation GELU (Gaussian Error Linear Unit).

    Formule: f(x) = x * Φ(x)
    où Φ(x) est la fonction de distribution cumulative normale

    Approximation: f(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name, trainable=False)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = inputs
        # Approximation de GELU
        sqrt_2_pi = np.sqrt(2 / np.pi)
        x_cubed = inputs ** 3
        inner = sqrt_2_pi * (inputs + 0.044715 * x_cubed)
        tanh_inner = np.tanh(inner)
        self.output = 0.5 * inputs * (1 + tanh_inner)
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        sqrt_2_pi = np.sqrt(2 / np.pi)
        x_cubed = self.input ** 3
        inner = sqrt_2_pi * (self.input + 0.044715 * x_cubed)
        tanh_inner = np.tanh(inner)
        derivative = 0.5 * (1 + tanh_inner) + 0.5 * self.input * (1 - tanh_inner ** 2) * sqrt_2_pi * (1 + 3 * 0.044715 * self.input ** 2)
        grad_input = grad_output * derivative
        return grad_input

    def __repr__(self) -> str:
        return "GELU()"
