# numpynet/optimizers/sgd.py
"""
SGD (Stochastic Gradient Descent)
=================================
Optimiseur de base avec option de momentum et decay.
Formule: w = w - lr * gradient
"""

import numpy as np
from typing import Optional, Any
from ..core.layer import Layer


class SGD:
    """
    SGD (Stochastic Gradient Descent).

    Formule de base: w = w - learning_rate * gradient

    Avec momentum:
        v = momentum * v - learning_rate * gradient
        w = w + v

    Attributes:
        learning_rate (float): Taux d'apprentissage.
        momentum (float): Coefficient de momentum.
        nesterov (bool): Si True, utilise Nesterov momentum.
        weight_decay (float): L2 regularization.
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 nesterov: bool = False,
                 weight_decay: float = 0.0,
                 name: Optional[str] = None):
        """
        Initialise l'optimiseur SGD.

        Args:
            learning_rate (float): Taux d'apprentissage.
            momentum (float): Coefficient de momentum.
            nesterov (bool): Si True, utilise Nesterov momentum.
            weight_decay (float): Decay L2 des poids.
        """
        self.name = name or "sgd"
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        # Accumulateurs de momentum (stockés par couche)
        self.velocity = {}  # {layer_id: [velocities]}

    def update(self, layer: Layer, grad_weights: np.ndarray, grad_bias: Optional[np.ndarray] = None) -> None:
        """
        Met à jour les poids de la couche.

        Args:
            layer (Layer): Couche à mettre à jour.
            grad_weights (np.ndarray): Gradient des poids.
            grad_bias (np.ndarray, optional): Gradient du biais.
        """
        if not layer.trainable:
            return

        layer_id = id(layer)

        # Initialiser le momentum si nécessaire
        if layer_id not in self.velocity:
            self.velocity[layer_id] = [
                np.zeros_like(layer.weights),
                np.zeros_like(layer.bias) if layer.bias is not None else None
            ]

        # Appliquer le weight decay (L2)
        if self.weight_decay > 0:
            grad_weights = grad_weights + self.weight_decay * layer.weights

        if self.momentum > 0:
            # Mise à jour avec momentum
            v_w = self.velocity[layer_id][0]

            if self.nesterov:
                # Nesterov momentum
                grad_weights = grad_weights + self.momentum * v_w
            else:
                # Classical momentum
                v_w = self.momentum * v_w - self.learning_rate * grad_weights

            # Mise à jour des poids
            layer.weights += v_w
            self.velocity[layer_id][0] = v_w

            if layer.bias is not None and grad_bias is not None:
                v_b = self.velocity[layer_id][1]
                if self.nesterov:
                    grad_bias = grad_bias + self.momentum * v_b
                else:
                    v_b = self.momentum * v_b - self.learning_rate * grad_bias

                layer.bias += v_b
                self.velocity[layer_id][1] = v_b
        else:
            # Mise à jour simple (vanilla SGD)
            layer.weights -= self.learning_rate * grad_weights
            if layer.bias is not None and grad_bias is not None:
                layer.bias -= self.learning_rate * grad_bias

    def get_config(self) -> dict:
        """Retourne la configuration de l'optimiseur."""
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'nesterov': self.nesterov,
            'weight_decay': self.weight_decay,
        }

    def __repr__(self) -> str:
        return f"SGD(lr={self.learning_rate}, momentum={self.momentum}, nesterov={self.nesterov})"


class GradientDescent:
    """
    Gradient Descent classique (par opposition à Stochastic).

    Utilise toute la batch pour chaque mise à jour.
    """

    def __init__(self, learning_rate: float = 0.01, name: Optional[str] = None):
        self.name = name or "gd"
        self.learning_rate = learning_rate

    def update(self, layer: Layer, grad_weights: np.ndarray, grad_bias: Optional[np.ndarray] = None) -> None:
        if not layer.trainable:
            return

        layer.weights -= self.learning_rate * grad_weights
        if layer.bias is not None and grad_bias is not None:
            layer.bias -= self.learning_rate * grad_bias

    def __repr__(self) -> str:
        return f"GradientDescent(lr={self.learning_rate})"
