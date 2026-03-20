# numpynet/activations/softmax.py
"""
Softmax
=======
Fonction d'activation utilisée pour la classification multiclasse.
Formule: f(x_i) = exp(x_i) / sum(exp(x_j)) pour tout j
"""

import numpy as np
from typing import Optional, Any
from ..core.layer import Layer


class Softmax(Layer):
    """
    Activation Softmax.

    Formule: softmax(x_i) = exp(x_i) / Σⱼ exp(x_j)

    Caractéristiques:
        - Somme des sorties = 1
        - Utilisée pour les probabilités en classification multiclasse
        - Sensible aux grandes valeurs (utiliser le shift numérique)

    Note: Pour la rétropropagation avec CrossEntropy, on peut simplifier
    en retournant directement (y_pred - y_true) car les dérivées se simplifient.
    """

    def __init__(self, axis: int = -1, name: Optional[str] = None):
        """
        Initialise l'activation Softmax.

        Args:
            axis (int): Axe sur lequel appliquer softmax.
            name (str, optional): Nom de la couche.
        """
        super().__init__(name=name, trainable=False)
        self.axis = axis

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant de Softmax.

        Args:
            inputs (np.ndarray): Tension d'entrée.
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Probabilités normalisées.
        """
        self.input = inputs

        # Shift pour la stabilité numérique (éviter overflow)
        inputs_shifted = inputs - np.max(inputs, axis=self.axis, keepdims=True)
        exp_inputs = np.exp(inputs_shifted)

        self.output = exp_inputs / np.sum(exp_inputs, axis=self.axis, keepdims=True)

        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière de Softmax.

        Note: L'implémentation complète de la Jacobienne peut être coûteuse.
        Pour l'utilisation typique avec CrossEntropy, cette méthode n'est pas
        directement appelée car la perte combine déjà les gradients.

        Args:
            grad_output (np.ndarray): Gradient de la perte par rapport à la sortie.
            optimizer (Optimizer, optional): Non utilisé.

        Returns:
            np.ndarray: Gradient de la perte par rapport à l'entrée.
        """
        # Version simplifiée: pour le cas typique softmax + crossentropy
        # le gradient est déjà calculé dans la fonction de perte

        # Version complète (plus coûteuse):
        batch_size = grad_output.shape[0]

        # Calculer la Jacobienne complète
        jacobian = np.zeros((batch_size, self.output.shape[-1], self.output.shape[-1]))

        for i in range(batch_size):
            s = self.output[i].reshape(-1, 1)
            jacobian[i] = np.diagflat(s) - np.dot(s, s.T)

        # Multiplier par le gradient
        grad_input = np.array([np.dot(jacobian[i], grad_output[i]) for i in range(batch_size)])

        return grad_input

    def gradient(self, grad_output: np.ndarray, output: np.ndarray) -> np.ndarray:
        """
        Calcule le gradient de Softmax (version simplifiée).

        Args:
            grad_output (np.ndarray): Gradient de la sortie.
            output (np.ndarray): Sortie de Softmax.

        Returns:
            np.ndarray: Gradient passé à la couche précédente.
        """
        # Note: Pour softmax + crossentropy, le gradient final est simplement:
        # grad = y_pred - y_true
        # Cette méthode est surtout utile pour d'autres combinaisons
        return grad_output

    def __repr__(self) -> str:
        return f"Softmax(axis={self.axis})"


class SoftmaxStable(Softmax):
    """
    Version alternative de Softmax avec stabilité numérique garantie.
    """

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.input = inputs
        # Méthode plus stable:shift de tous les inputs
        exp_x = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.output
