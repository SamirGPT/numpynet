# numpynet/optimizers/adam.py
"""
Adam (Adaptive Moment Estimation)
=================================
Optimiseur populaire avec estimation adaptative des moments.
Combine les avantages de RMSprop et Momentum.
"""

import numpy as np
from typing import Optional
from ..core.layer import Layer


class Adam:
    """
    Adam (Adaptive Moment Estimation).

    Algorithme:
        m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
        v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)

    Avantages:
        - Adaptatif (différents taux d'apprentissage par paramètre)
        - Bon pour la plupart des problèmes
        - Peut fonctionner sans tuning fin

    References:
        - Kingma, D. P., & Ba, J. L. (2014). Adam: A method for stochastic optimization.
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 amsgrad: bool = False,
                 weight_decay: float = 0.0,
                 name: Optional[str] = None):
        """
        Initialise l'optimiseur Adam.

        Args:
            learning_rate (float): Taux d'apprentissage.
            beta1 (float): Taux de décroissance du premier moment.
            beta2 (float): Taux de décroissance du deuxième moment.
            epsilon (float): Petit terme pour la stabilité numérique.
            amsgrad (bool): Si True, utilise AMSGrad.
            weight_decay (float): Decay L2 des poids.
        """
        self.name = name or "adam"
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.weight_decay = weight_decay

        # Compteur d'itérations
        self.iterations = 0

        # Accumulateurs (stockés par couche)
        self.m = {}   # Premier moment (moyenne du gradient)
        self.v = {}   # Deuxième moment (variance)
        self.v_hat = {}  # Pour AMSGrad

    def update(self, layer: Layer, grad_weights: np.ndarray, grad_bias: Optional[np.ndarray] = None) -> None:
        """
        Met à jour les poids de la couche avec Adam.

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
            if self.amsgrad:
                self.v_hat[layer_id] = [
                    np.zeros_like(layer.weights),
                    np.zeros_like(layer.bias) if layer.bias is not None else None
                ]

        # Facteur de biais
        bias_correction1 = 1 - self.beta1 ** self.iterations
        bias_correction2 = 1 - self.beta2 ** self.iterations

        # Mise à jour des poids
        m_w, v_w = self.m[layer_id]
        m_b, v_b = self.m[layer_id][1], self.v[layer_id][1]

        # Appliquer le weight decay (L2)
        if self.weight_decay > 0:
            grad_weights = grad_weights + self.weight_decay * layer.weights

        # Mise à jour du premier moment
        m_w = self.beta1 * m_w + (1 - self.beta1) * grad_weights
        self.m[layer_id][0] = m_w

        # Mise à jour du deuxième moment
        v_w = self.beta2 * v_w + (1 - self.beta2) * (grad_weights ** 2)
        self.v[layer_id][0] = v_w

        # Correction du biais
        m_w_hat = m_w / bias_correction1
        v_w_hat = v_w / bias_correction2

        # AMSGrad
        if self.amsgrad:
            self.v_hat[layer_id][0] = np.maximum(self.v_hat[layer_id][0], v_w_hat)
            v_w_hat = self.v_hat[layer_id][0]

        # Mise à jour des poids
        layer.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        # Mise à jour du biais si présent
        if layer.bias is not None and grad_bias is not None:
            # Appliquer le weight decay
            if self.weight_decay > 0:
                grad_bias = grad_bias + self.weight_decay * layer.bias

            # Premier moment
            m_b = self.beta1 * m_b + (1 - self.beta1) * grad_bias
            self.m[layer_id][1] = m_b

            # Deuxième moment
            v_b = self.beta2 * v_b + (1 - self.beta2) * (grad_bias ** 2)
            self.v[layer_id][1] = v_b

            # Correction du biais
            m_b_hat = m_b / bias_correction1
            v_b_hat = v_b / bias_correction2

            if self.amsgrad:
                self.v_hat[layer_id][1] = np.maximum(self.v_hat[layer_id][1], v_b_hat)
                v_b_hat = self.v_hat[layer_id][1]

            layer.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def get_config(self) -> dict:
        """Retourne la configuration de l'optimiseur."""
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'weight_decay': self.weight_decay,
        }

    def __repr__(self) -> str:
        return f"Adam(lr={self.learning_rate}, beta1={self.beta1}, beta2={self.beta2})"


class Adamax:
    """
    Adamax - Variante d'Adam utilisant la norme L-infinie.
    """

    def __init__(self,
                 learning_rate: float = 0.002,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 name: Optional[str] = None):
        self.name = name or "adamax"
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

        self.m = {}
        self.u = {}

    def update(self, layer: Layer, grad_weights: np.ndarray, grad_bias: Optional[np.ndarray] = None) -> None:
        if not layer.trainable:
            return

        layer_id = id(layer)
        self.iterations += 1

        if layer_id not in self.m:
            self.m[layer_id] = [np.zeros_like(layer.weights), np.zeros_like(layer.bias) if layer.bias is not None else None]
            self.u[layer_id] = [np.zeros_like(layer.weights), np.zeros_like(layer.bias) if layer.bias is not None else None]

        bias_correction1 = 1 - self.beta1 ** self.iterations

        # Mise à jour des poids
        m_w = self.m[layer_id][0]
        m_w = self.beta1 * m_w + (1 - self.beta1) * grad_weights
        self.m[layer_id][0] = m_w

        u_w = np.maximum(self.beta2 * self.u[layer_id][0], np.abs(grad_weights))
        self.u[layer_id][0] = u_w

        layer.weights -= (self.learning_rate / bias_correction1) * m_w / (u_w + self.epsilon)

        if layer.bias is not None and grad_bias is not None:
            m_b = self.beta1 * self.m[layer_id][1] + (1 - self.beta1) * grad_bias
            self.m[layer_id][1] = m_b
            u_b = np.maximum(self.beta2 * self.u[layer_id][1], np.abs(grad_bias))
            self.u[layer_id][1] = u_b
            layer.bias -= (self.learning_rate / bias_correction1) * m_b / (u_b + self.epsilon)

    def __repr__(self) -> str:
        return f"Adamax(lr={self.learning_rate})"


class Nadam:
    """
    NAdam - Combine Nesterov momentum avec Adam.
    """

    def __init__(self,
                 learning_rate: float = 0.002,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 name: Optional[str] = None):
        self.name = name or "nadam"
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0
        self.m = {}
        self.v = {}

    def update(self, layer: Layer, grad_weights: np.ndarray, grad_bias: Optional[np.ndarray] = None) -> None:
        if not layer.trainable:
            return

        layer_id = id(layer)
        self.iterations += 1

        if layer_id not in self.m:
            self.m[layer_id] = [np.zeros_like(layer.weights), np.zeros_like(layer.bias) if layer.bias is not None else None]
            self.v[layer_id] = [np.zeros_like(layer.weights), np.zeros_like(layer.bias) if layer.bias is not None else None]

        bias_correction1 = 1 - self.beta1 ** self.iterations
        bias_correction2 = 1 - self.beta2 ** self.iterations

        # Nesterov
        m_w = self.beta1 * self.m[layer_id][0] + (1 - self.beta1) * grad_weights
        self.m[layer_id][0] = m_w

        v_w = self.beta2 * self.v[layer_id][0] + (1 - self.beta2) * (grad_weights ** 2)
        self.v[layer_id][0] = v_w

        m_w_hat = (m_w * self.beta1) / bias_correction1 + (1 - self.beta1) * grad_weights / bias_correction1
        v_w_hat = v_w / bias_correction2

        layer.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        if layer.bias is not None and grad_bias is not None:
            m_b = self.beta1 * self.m[layer_id][1] + (1 - self.beta1) * grad_bias
            self.m[layer_id][1] = m_b
            v_b = self.beta2 * self.v[layer_id][1] + (1 - self.beta2) * (grad_bias ** 2)
            self.v[layer_id][1] = v_b

            m_b_hat = (m_b * self.beta1) / bias_correction1 + (1 - self.beta1) * grad_bias / bias_correction1
            v_b_hat = v_b / bias_correction2

            layer.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def __repr__(self) -> str:
        return f"Nadam(lr={self.learning_rate})"
