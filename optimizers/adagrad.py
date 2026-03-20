# numpynet/optimizers/adagrad.py
"""
Adagrad
=======
Optimiseur avec taux d'apprentissage adaptatif basé sur l'historique des gradients.
"""

import numpy as np
from typing import Optional
from ..core.layer import Layer


class Adagrad:
    """
    Adagrad (Adaptive Gradient Algorithm).

    Algorithme:
        G = G + gradient^2
        w = w - learning_rate * gradient / sqrt(G + epsilon)

    Avantages:
        - Adaptatif: différent lr par paramètre
        - Bon pour les données clairsemées (sparse)
    Inconvénients:
        - Le lr diminue monotonic au fil du temps
        - Peut arrêter l'entraînement trop tôt
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 epsilon: float = 1e-7,
                 name: Optional[str] = None):
        """
        Initialise l'optimiseur Adagrad.

        Args:
            learning_rate (float): Taux d'apprentissage initial.
            epsilon (float): Petit terme pour la stabilité numérique.
        """
        self.name = name or "adagrad"
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # Accumulateurs (stockés par couche)
        self.G = {}  # Somme cumulative des carrés des gradients

    def update(self, layer: Layer, grad_weights: np.ndarray, grad_bias: Optional[np.ndarray] = None) -> None:
        """
        Met à jour les poids de la couche avec Adagrad.

        Args:
            layer (Layer): Couche à mettre à jour.
            grad_weights (np.ndarray): Gradient des poids.
            grad_bias (np.ndarray, optional): Gradient du biais.
        """
        if not layer.trainable:
            return

        layer_id = id(layer)

        # Initialiser les accumulateurs si nécessaire
        if layer_id not in self.G:
            self.G[layer_id] = [
                np.zeros_like(layer.weights),
                np.zeros_like(layer.bias) if layer.bias is not None else None
            ]

        # Mise à jour des accumulateurs
        G_w = self.G[layer_id][0]
        G_w = G_w + grad_weights ** 2
        self.G[layer_id][0] = G_w

        # Mise à jour des poids
        layer.weights -= self.learning_rate * grad_weights / np.sqrt(G_w + self.epsilon)

        # Mise à jour du biais
        if layer.bias is not None and grad_bias is not None:
            G_b = self.G[layer_id][1]
            G_b = G_b + grad_bias ** 2
            self.G[layer_id][1] = G_b

            layer.bias -= self.learning_rate * grad_bias / np.sqrt(G_b + self.epsilon)

    def get_config(self) -> dict:
        """Retourne la configuration de l'optimiseur."""
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
        }

    def __repr__(self) -> str:
        return f"Adagrad(lr={self.learning_rate})"
