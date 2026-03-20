# numpynet/optimizers/momentum.py
"""
Momentum
========
Optimiseur avec accélération par le moment.
"""

import numpy as np
from typing import Optional
from ..core.layer import Layer


class Momentum:
    """
    Momentum (Classical Momentum).

    Algorithme:
        v = momentum * v - learning_rate * gradient
        w = w + v

    Avantages:
        - Accélère la convergence
        - Réduit les oscillations
        - Peut sortir des minima locaux

    Variantes:
        - Classical Momentum: direction précédente
        - Nesterov: regarde ahead avant de calculer le gradient
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 nesterov: bool = False,
                 name: Optional[str] = None):
        """
        Initialise l'optimiseur Momentum.

        Args:
            learning_rate (float): Taux d'apprentissage.
            momentum (float): Coefficient de momentum.
            nesterov (bool): Si True, utilise Nesterov momentum.
        """
        self.name = name or "momentum"
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

        # Accumulateurs de vitesse (stockés par couche)
        self.velocity = {}

    def update(self, layer: Layer, grad_weights: np.ndarray, grad_bias: Optional[np.ndarray] = None) -> None:
        """
        Met à jour les poids de la couche avec Momentum.

        Args:
            layer (Layer): Couche à mettre à jour.
            grad_weights (np.ndarray): Gradient des poids.
            grad_bias (np.ndarray, optional): Gradient du biais.
        """
        if not layer.trainable:
            return

        layer_id = id(layer)

        # Initialiser la vitesse si nécessaire
        if layer_id not in self.velocity:
            self.velocity[layer_id] = [
                np.zeros_like(layer.weights),
                np.zeros_like(layer.bias) if layer.bias is not None else None
            ]

        v_w = self.velocity[layer_id][0]

        if self.nesterov:
            # Nesterov: regardeahead
            grad_weights = grad_weights + self.momentum * v_w
        else:
            # Classical momentum
            v_w = self.momentum * v_w - self.learning_rate * grad_weights

        # Mise à jour des poids
        layer.weights += v_w
        self.velocity[layer_id][0] = v_w

        # Mise à jour du biais
        if layer.bias is not None and grad_bias is not None:
            v_b = self.velocity[layer_id][1]

            if self.nesterov:
                grad_bias = grad_bias + self.momentum * v_b
            else:
                v_b = self.momentum * v_b - self.learning_rate * grad_bias

            layer.bias += v_b
            self.velocity[layer_id][1] = v_b

    def get_config(self) -> dict:
        """Retourne la configuration de l'optimiseur."""
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'nesterov': self.nesterov,
        }

    def __repr__(self) -> str:
        return f"Momentum(lr={self.learning_rate}, momentum={self.momentum}, nesterov={self.nesterov})"
