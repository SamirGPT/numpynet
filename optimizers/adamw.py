# numpynet/optimizers/adamw.py
"""
AdamW (Adam with Weight Decay)
==============================
Optimiseur Adam avec découplage du weight decay.
"""

import numpy as np
from typing import Optional
from ..core.layer import Layer


class AdamW:
    """
    AdamW (Adam with Weight Decay).

    Différence avec Adam:
    - Le weight decay est appliqué APRÈS la mise à jour du lr
    - Cela découple effectivement le weight decay de l'estimation du gradient

    Algorithme:
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        w = w - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * w

    References:
        - Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization.
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.01,
                 name: Optional[str] = None):
        """
        Initialise l'optimiseur AdamW.

        Args:
            learning_rate (float): Taux d'apprentissage.
            beta1 (float): Taux de décroissance du premier moment.
            beta2 (float): Taux de décroissance du deuxième moment.
            epsilon (float): Petit terme pour la stabilité numérique.
            weight_decay (float): Coefficient de weight decay (L2 regularisation découplée).
        """
        self.name = name or "adamw"
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        # Compteur d'itérations
        self.iterations = 0

        # Accumulateurs (stockés par couche)
        self.m = {}
        self.v = {}

    def update(self, layer: Layer, grad_weights: np.ndarray, grad_bias: Optional[np.ndarray] = None) -> None:
        """
        Met à jour les poids de la couche avec AdamW.

        Args:
            layer (Layer): Couche à mettre à jour.
            grad_weights (np.ndarray): Gradient des poids.
            grad_bias (np.ndarray, optional): Gradient du biais.
        """
        if not layer.trainable:
            return

        layer_id = id(layer)
        self.iterations += 1

        # Initialiser les accumulateurs si nécessaire
        if layer_id not in self.m:
            self.m[layer_id] = [
                np.zeros_like(layer.weights),
                np.zeros_like(layer.bias) if layer.bias is not None else None
            ]
            self.v[layer_id] = [
                np.zeros_like(layer.weights),
                np.zeros_like(layer.bias) if layer.bias is not None else None
            ]

        # Facteur de biais
        bias_correction1 = 1 - self.beta1 ** self.iterations
        bias_correction2 = 1 - self.beta2 ** self.iterations

        # Mise à jour du premier moment (gradient)
        m_w = self.beta1 * self.m[layer_id][0] + (1 - self.beta1) * grad_weights
        self.m[layer_id][0] = m_w

        # Mise à jour du deuxième moment (carré du gradient)
        v_w = self.beta2 * self.v[layer_id][0] + (1 - self.beta2) * (grad_weights ** 2)
        self.v[layer_id][0] = v_w

        # Correction du biais
        m_w_hat = m_w / bias_correction1
        v_w_hat = v_w / bias_correction2

        # Mise à jour des poids avec weight decay
        # Note: weight decay est appliqué APRÈS la mise à jour du gradient (découplé)
        layer.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        layer.weights -= self.learning_rate * self.weight_decay * layer.weights

        # Mise à jour du biais
        if layer.bias is not None and grad_bias is not None:
            m_b = self.beta1 * self.m[layer_id][1] + (1 - self.beta1) * grad_bias
            self.m[layer_id][1] = m_b

            v_b = self.beta2 * self.v[layer_id][1] + (1 - self.beta2) * (grad_bias ** 2)
            self.v[layer_id][1] = v_b

            m_b_hat = m_b / bias_correction1
            v_b_hat = v_b / bias_correction2

            layer.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def get_config(self) -> dict:
        """Retourne la configuration de l'optimiseur."""
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
        }

    def __repr__(self) -> str:
        return f"AdamW(lr={self.learning_rate}, weight_decay={self.weight_decay})"
