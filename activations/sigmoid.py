# numpynet/activations/sigmoid.py
"""
Sigmoid (Logistic Function)
===========================
Fonction d'activation très utilisée pour la classification binaire.
Formule: f(x) = 1 / (1 + exp(-x))
Dérivée: f'(x) = f(x) * (1 - f(x))
"""

import numpy as np
from typing import Optional, Any
from ..core.layer import Layer


class Sigmoid(Layer):
    """
    Activation Sigmoid (Fonction Logistique).

    Formule: σ(x) = 1 / (1 + e^(-x))
    Dérivée: σ'(x) = σ(x) * (1 - σ(x))

    Caractéristiques:
        - Sortie entre 0 et 1
        - Utilisée pour les probabilités
        - Problème de vanishing gradient pour les grandes valeurs positives/négatives
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialise l'activation Sigmoid.

        Args:
            name (str, optional): Nom de la couche.
        """
        super().__init__(name=name, trainable=False)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant de Sigmoid.

        Args:
            inputs (np.ndarray): Tension d'entrée.
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Sortie de Sigmoid (entre 0 et 1).
        """
        self.input = inputs
        # Pour la stabilité numérique, on utilise clip
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -500, 500)))
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière de Sigmoid.

        Args:
            grad_output (np.ndarray): Gradient de la perte par rapport à la sortie.
            optimizer (Optimizer, optional): Non utilisé.

        Returns:
            np.ndarray: Gradient de la perte par rapport à l'entrée.
        """
        # Dérivée: σ'(x) = σ(x) * (1 - σ(x))
        sigmoid_derivative = self.output * (1 - self.output)
        grad_input = grad_output * sigmoid_derivative
        return grad_input

    def gradient(self, grad_output: np.ndarray, output: np.ndarray) -> np.ndarray:
        """
        Calcule le gradient de Sigmoid.

        Args:
            grad_output (np.ndarray): Gradient de la sortie.
            output (np.ndarray): Sortie de Sigmoid.

        Returns:
            np.ndarray: Gradient passé à la couche précédente.
        """
        return grad_output * output * (1 - output)

    def __repr__(self) -> str:
        return "Sigmoid()"


class HardSigmoid(Layer):
    """
    Activation Hard Sigmoid - Approximation linéaire de Sigmoid.

    Formule: f(x) = clip(0.2*x + 0.5, 0, 1)
    Plus rapide que Sigmoid car pas d'exponentielle.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name, trainable=False)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = inputs
        self.output = np.clip(0.2 * inputs + 0.5, 0, 1)
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        # Dérivée: 0.2 si 0 < output < 1, sinon 0
        grad_input = grad_output * ((self.output > 0) & (self.output < 1)).astype(float) * 0.2
        return grad_input

    def __repr__(self) -> str:
        return "HardSigmoid()"
