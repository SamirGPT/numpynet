# numpynet/activations/elu.py
"""
ELU (Exponential Linear Unit)
==============================
Variante de ReLU avec une exponentielle pour les valeurs négatives.
Formule: f(x) = x si x > 0, sinon alpha * (exp(x) - 1)
Dérivée: f'(x) = 1 si x > 0, sinon f(x) + alpha
"""

import numpy as np
from typing import Optional, Any
from ..core.layer import Layer


class ELU(Layer):
    """
    Activation ELU (Exponential Linear Unit).

    Formule: f(x) = x si x > 0, sinon α(e^x - 1)
    Dérivée: f'(x) = 1 si x > 0, sinon f(x) + α

    Avantages:
        - Sortie proche de 0 (meilleure distribution)
        - gradients non nuls pour les valeurs négatives
        - Meilleure convergence que ReLU

    Inconvénients:
        - Calcul plus coûteux (exponentielle)
    """

    def __init__(self, alpha: float = 1.0, name: Optional[str] = None):
        """
        Initialise l'activation ELU.

        Args:
            alpha (float): Paramètre d'échelle pour les valeurs négatives.
            name (str, optional): Nom de la couche.
        """
        super().__init__(name=name, trainable=False)
        self.alpha = alpha

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant de ELU.

        Args:
            inputs (np.ndarray): Tension d'entrée.
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Sortie de ELU.
        """
        self.input = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * (np.exp(inputs) - 1))
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière de ELU.

        Args:
            grad_output (np.ndarray): Gradient de la perte par rapport à la sortie.
            optimizer (Optimizer, optional): Non utilisé.

        Returns:
            np.ndarray: Gradient de la perte par rapport à l'entrée.
        """
        # Dérivée: 1 si x > 0, sinon f(x) + alpha
        grad_input = np.where(
            self.input > 0,
            grad_output,
            grad_output * (self.output + self.alpha)
        )
        return grad_input

    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha})"


class SELU(Layer):
    """
    Activation SELU (Scaled Exponential Linear Unit).

    Variante auto-normalisée de ELU avec des paramètres spécifiques.
    Pour des entrées de moyenne 0 et variance 1, la sortie est normalisée.

    Note: Cette activation nécessite:
    - Une initialisation spécifique (LeCun)
    - La normalisation des entrées
    """

    def __init__(self, name: Optional[str] = None):
        # Constantes pour SELU
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        super().__init__(name=name, trainable=False)
        self.alpha = alpha
        self.scale = scale

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = inputs
        self.output = self.scale * np.where(
            inputs > 0,
            inputs,
            self.alpha * (np.exp(inputs) - 1)
        )
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        # Dérivée avec le scale
        grad_input = self.scale * np.where(
            self.input > 0,
            grad_output,
            grad_output * (self.output / self.scale + self.alpha)
        )
        return grad_input

    def __repr__(self) -> str:
        return "SELU()"
