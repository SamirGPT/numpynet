# numpynet/activations/tanh.py
"""
Tanh (Hyperbolic Tangent)
=========================
Fonction d'activation dont la sortie est centrée sur 0.
Formule: f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
Dérivée: f'(x) = 1 - f(x)²
"""

import numpy as np
from typing import Optional, Any
from ..core.layer import Layer


class Tanh(Layer):
    """
    Activation Tanh (Tangente Hyperbolique).

    Formule: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Dérivée: tanh'(x) = 1 - tanh²(x)

    Caractéristiques:
        - Sortie entre -1 et 1
        - Moyenne à 0 (meilleur que sigmoid pour les couches cachées)
        - Problème de vanishing gradient pour les grandes valeurs
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialise l'activation Tanh.

        Args:
            name (str, optional): Nom de la couche.
        """
        super().__init__(name=name, trainable=False)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant de Tanh.

        Args:
            inputs (np.ndarray): Tension d'entrée.
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Sortie de Tanh (entre -1 et 1).
        """
        self.input = inputs
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière de Tanh.

        Args:
            grad_output (np.ndarray): Gradient de la perte par rapport à la sortie.
            optimizer (Optimizer, optional): Non utilisé.

        Returns:
            np.ndarray: Gradient de la perte par rapport à l'entrée.
        """
        # Dérivée: tanh'(x) = 1 - tanh²(x)
        tanh_derivative = 1 - np.square(self.output)
        grad_input = grad_output * tanh_derivative
        return grad_input

    def gradient(self, grad_output: np.ndarray, output: np.ndarray) -> np.ndarray:
        """
        Calcule le gradient de Tanh.

        Args:
            grad_output (np.ndarray): Gradient de la sortie.
            output (np.ndarray): Sortie de Tanh.

        Returns:
            np.ndarray: Gradient passé à la couche précédente.
        """
        return grad_output * (1 - np.square(output))

    def __repr__(self) -> str:
        return "Tanh()"


class TanhShrink(Layer):
    """
    Activation Tanh Shrink.

    Formule: f(x) = x - tanh(x)
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name, trainable=False)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = inputs
        self.output = inputs - np.tanh(inputs)
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        # Dérivée: 1 - tanh'(x) = 1 - (1 - tanh²(x)) = tanh²(x)
        grad_input = grad_output * np.square(np.tanh(self.input))
        return grad_input

    def __repr__(self) -> str:
        return "TanhShrink()"
