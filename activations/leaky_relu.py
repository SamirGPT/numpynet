# numpynet/activations/leaky_relu.py
"""
Leaky ReLU
==========
Variante de ReLU qui permet un petit gradient pour les valeurs négatives.
Formule: f(x) = x si x > 0, sinon alpha * x
Dérivée: f'(x) = 1 si x > 0, sinon alpha
"""

import numpy as np
from typing import Optional, Any
from ..core.layer import Layer


class LeakyReLU(Layer):
    """
    Activation Leaky ReLU.

    Formule: f(x) = max(αx, x) où α est typiquement 0.01
    Dérivée: f'(x) = 1 si x > 0, sinon α

    Avantages:
        - Évite le problème des "dying ReLU"
        -_gradient non nul pour les entrées négatives
    """

    def __init__(self, alpha: float = 0.01, name: Optional[str] = None):
        """
        Initialise l'activation LeakyReLU.

        Args:
            alpha (float): Pente pour les valeurs négatives.
            name (str, optional): Nom de la couche.
        """
        super().__init__(name=name, trainable=False)
        self.alpha = alpha

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant de LeakyReLU.

        Args:
            inputs (np.ndarray): Tension d'entrée.
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Sortie de LeakyReLU.
        """
        self.input = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière de LeakyReLU.

        Args:
            grad_output (np.ndarray): Gradient de la perte par rapport à la sortie.
            optimizer (Optimizer, optional): Non utilisé.

        Returns:
            np.ndarray: Gradient de la perte par rapport à l'entrée.
        """
        grad_input = np.where(self.input > 0, grad_output, self.alpha * grad_output)
        return grad_input

    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


class PReLU(Layer):
    """
    Activation PReLU (Parametric ReLU).

    Variante de LeakyReLU où le paramètre alpha est apprenable.
    """

    def __init__(self, alpha: float = 0.01, name: Optional[str] = None):
        super().__init__(name=name, trainable=True)
        self.alpha = np.array(alpha)
        self._alpha = None  # Pour stocker alpha pendant le forward

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = inputs
        self._alpha = self.alpha
        if self._alpha.ndim < inputs.ndim:
            self._alpha = self._alpha.reshape(1, -1)
        self.output = np.where(inputs > 0, inputs, self._alpha * inputs)
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        grad_input = np.where(self.input > 0, grad_output, self._alpha * grad_output)
        # Gradient pour alpha (simplifié)
        grad_alpha = np.sum(np.where(self.input < 0, self.input * grad_output, 0))

        # Mise à jour de alpha
        if optimizer is not None:
            self.alpha -= optimizer.learning_rate * grad_alpha
            self.alpha = np.clip(self.alpha, 0, 1)

        return grad_input

    def __repr__(self) -> str:
        return f"PReLU(alpha={self.alpha})"


class RReLU(Layer):
    """
    Activation RReLU (Randomized ReLU).

    Le alpha est samplé aléatoirement pendant l'entraînement.
    """

    def __init__(self, lower: float = 1/8, upper: float = 1/3, name: Optional[str] = None):
        super().__init__(name=name, trainable=False)
        self.lower = lower
        self.upper = upper
        self.alpha = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = inputs
        if training:
            # Sample alpha aléatoirement pendant l'entraînement
            self.alpha = np.random.uniform(self.lower, self.upper)
        else:
            # Moyenne pendant l'inférence
            self.alpha = (self.lower + self.upper) / 2

        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        return np.where(self.input > 0, grad_output, self.alpha * grad_output)

    def __repr__(self) -> str:
        return f"RReLU(lower={self.lower}, upper={self.upper})"
