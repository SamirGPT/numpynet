# numpynet/optimizers/rmsprop.py
"""
RMSprop
=======
Optimiseur avec taux d'apprentissage adaptatif.
Divise le taux d'apprentissage par une moyenne mobile de l'amplitude des gradients.
"""

import numpy as np
from typing import Optional
from ..core.layer import Layer


class RMSprop:
    """
    RMSprop (Root Mean Square Propagation).

    Algorithme:
        v = rho * v + (1 - rho) * gradient^2
        w = w - learning_rate * gradient / sqrt(v + epsilon)

    Avantages:
        - Adaptatif: différent lr par paramètre
        - Bon pour les réseaux récurrents
        - Fonctionne bien sans tuning fin

    References:
        - Hinton, G. (2012). Lecture 6.5 - RMSProp.
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 rho: float = 0.9,
                 momentum: float = 0.0,
                 epsilon: float = 1e-7,
                 centered: bool = False,
                 weight_decay: float = 0.0,
                 name: Optional[str] = None):
        """
        Initialise l'optimiseur RMSprop.

        Args:
            learning_rate (float): Taux d'apprentissage.
            rho (float): Taux de décroissance de la moyenne mobile.
            momentum (float): Coefficient de momentum.
            epsilon (float): Petit terme pour la stabilité numérique.
            centered (bool): Si True, centre la variance.
            weight_decay (float): Decay L2 des poids.
        """
        self.name = name or "rmsprop"
        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered
        self.weight_decay = weight_decay

        # Accumulateurs (stockés par couche)
        self.v = {}  # Moyenne mobile du carré des gradients
        self.g = {}  # Moyenne mobile des gradients (pour centered)
        self.m = {}  # Moment (pour momentum)

    def update(self, layer: Layer, grad_weights: np.ndarray, grad_bias: Optional[np.ndarray] = None) -> None:
        """
        Met à jour les poids de la couche avec RMSprop.

        Args:
            layer (Layer): Couche à mettre à jour.
            grad_weights (np.ndarray): Gradient des poids.
            grad_bias (np.ndarray, optional): Gradient du biais.
        """
        if not layer.trainable:
            return

        layer_id = id(layer)

        # Initialiser les accumulateurs si nécessaire
        if layer_id not in self.v:
            self.v[layer_id] = [
                np.zeros_like(layer.weights),
                np.zeros_like(layer.bias) if layer.bias is not None else None
            ]
            if self.centered:
                self.g[layer_id] = [
                    np.zeros_like(layer.weights),
                    np.zeros_like(layer.bias) if layer.bias is not None else None
                ]
            if self.momentum > 0:
                self.m[layer_id] = [
                    np.zeros_like(layer.weights),
                    np.zeros_like(layer.bias) if layer.bias is not None else None
                ]

        # Appliquer le weight decay (L2)
        if self.weight_decay > 0:
            grad_weights = grad_weights + self.weight_decay * layer.weights

        # Mise à jour de la variance
        v_w = self.v[layer_id][0]
        v_w = self.rho * v_w + (1 - self.rho) * (grad_weights ** 2)
        self.v[layer_id][0] = v_w

        # Calcul du denominateur
        if self.centered:
            g_w = self.g[layer_id][0]
            g_w = self.rho * g_w + (1 - self.rho) * grad_weights
            self.g[layer_id][0] = g_w
            denom = np.sqrt(v_w - g_w ** 2 + self.epsilon)
        else:
            denom = np.sqrt(v_w + self.epsilon)

        # Mise à jour avec ou sans momentum
        if self.momentum > 0:
            m_w = self.m[layer_id][0]
            m_w = self.momentum * m_w - self.learning_rate * grad_weights / denom
            self.m[layer_id][0] = m_w
            layer.weights += m_w
        else:
            layer.weights -= self.learning_rate * grad_weights / denom

        # Mise à jour du biais
        if layer.bias is not None and grad_bias is not None:
            if self.weight_decay > 0:
                grad_bias = grad_bias + self.weight_decay * layer.bias

            v_b = self.rho * self.v[layer_id][1] + (1 - self.rho) * (grad_bias ** 2)
            self.v[layer_id][1] = v_b

            if self.centered:
                g_b = self.rho * self.g[layer_id][1] + (1 - self.rho) * grad_bias
                self.g[layer_id][1] = g_b
                denom = np.sqrt(v_b - g_b ** 2 + self.epsilon)
            else:
                denom = np.sqrt(v_b + self.epsilon)

            if self.momentum > 0:
                m_b = self.momentum * self.m[layer_id][1] - self.learning_rate * grad_bias / denom
                self.m[layer_id][1] = m_b
                layer.bias += m_b
            else:
                layer.bias -= self.learning_rate * grad_bias / denom

    def get_config(self) -> dict:
        """Retourne la configuration de l'optimiseur."""
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'rho': self.rho,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'centered': self.centered,
            'weight_decay': self.weight_decay,
        }

    def __repr__(self) -> str:
        return f"RMSprop(lr={self.learning_rate}, rho={self.rho})"


class Adadelta:
    """
    Adadelta - Extension de Adagrad avecdecay des accumulateurs.
    """

    def __init__(self,
                 learning_rate: float = 1.0,
                 rho: float = 0.95,
                 epsilon: float = 1e-7,
                 name: Optional[str] = None):
        self.name = name or "adadelta"
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

        self.accumulated_grads = {}
        self.accumulated_deltas = {}

    def update(self, layer: Layer, grad_weights: np.ndarray, grad_bias: Optional[np.ndarray] = None) -> None:
        if not layer.trainable:
            return

        layer_id = id(layer)

        if layer_id not in self.accumulated_grads:
            self.accumulated_grads[layer_id] = [
                np.zeros_like(layer.weights),
                np.zeros_like(layer.bias) if layer.bias is not None else None
            ]
            self.accumulated_deltas[layer_id] = [
                np.zeros_like(layer.weights),
                np.zeros_like(layer.bias) if layer.bias is not None else None
            ]

        # Mise à jour des accumulateurs de gradients
        acc_grad_w = self.accumulated_grads[layer_id][0]
        acc_grad_w = self.rho * acc_grad_w + (1 - self.rho) * (grad_weights ** 2)
        self.accumulated_grads[layer_id][0] = acc_grad_w

        # Calcul du delta
        delta_w = -np.sqrt(acc_grad_w + self.epsilon) / np.sqrt(grad_weights ** 2 + self.epsilon) * grad_weights
        delta_w = -self.learning_rate * delta_w
        layer.weights += delta_w

        # Mise à jour des accumulateurs de deltas
        acc_delta_w = self.accumulated_deltas[layer_id][0]
        acc_delta_w = self.rho * acc_delta_w + (1 - self.rho) * (delta_w ** 2)
        self.accumulated_deltas[layer_id][0] = acc_delta_w

        if layer.bias is not None and grad_bias is not None:
            acc_grad_b = self.rho * self.accumulated_grads[layer_id][1] + (1 - self.rho) * (grad_bias ** 2)
            self.accumulated_grads[layer_id][1] = acc_grad_b

            delta_b = -self.learning_rate * grad_bias * np.sqrt(acc_grad_b + self.epsilon) / np.sqrt(grad_bias ** 2 + self.epsilon)
            layer.bias += delta_b

            acc_delta_b = self.rho * self.accumulated_deltas[layer_id][1] + (1 - self.rho) * (delta_b ** 2)
            self.accumulated_deltas[layer_id][1] = acc_delta_b

    def __repr__(self) -> str:
        return f"Adadelta(lr={self.learning_rate}, rho={self.rho})"
