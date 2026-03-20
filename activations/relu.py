# numpynet/activations/relu.py
"""
ReLU (Rectified Linear Unit)
=============================
Fonction d'activation la plus utilisée dans les réseaux de neurones modernes.
Formule: f(x) = max(0, x)
Dérivée: f'(x) = 1 si x > 0, sinon 0
"""

import numpy as np
from typing import Optional, Any
from ..core.layer import Layer


class ReLU(Layer):
    """
    Activation ReLU (Rectified Linear Unit).

    Fonction: f(x) = max(0, x)
    Dérivée: f'(x) = 1 si x > 0, sinon 0

    Avantages:
        - Calcul simple et rapide
        - Réduit le problème de vanishing gradient
        - Introduce la sparsité dans le réseau

    Inconvénients:
        - Problème des "dying ReLU" (neurones qui meurent pendant l'entraînement)
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialise l'activation ReLU.

        Args:
            name (str, optional): Nom de la couche.
        """
        super().__init__(name=name, trainable=False)

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant de ReLU.

        Args:
            inputs (np.ndarray): Tension d'entrée.
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Sortie de ReLU.
        """
        self.input = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière de ReLU.

        Args:
            grad_output (np.ndarray): Gradient de la perte par rapport à la sortie.
            optimizer (Optimizer, optional): Non utilisé (pas de paramètres).

        Returns:
            np.ndarray: Gradient de la perte par rapport à l'entrée.
        """
        # Dérivée: grad_input = grad_output * (input > 0)
        grad_input = grad_output * (self.input > 0).astype(float)
        return grad_input

    def gradient(self, grad_output: np.ndarray, output: np.ndarray) -> np.ndarray:
        """
        Calcule le gradient de l'activation (pour usage interne).

        Args:
            grad_output (np.ndarray): Gradient de la sortie.
            output (np.ndarray): Sortie de la couche (pour éviter recalcul).

        Returns:
            np.ndarray: Gradient passé à la couche précédente.
        """
        return grad_output * (output > 0).astype(float)

    def __repr__(self) -> str:
        return "ReLU()"
