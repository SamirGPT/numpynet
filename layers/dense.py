# numpynet/layers/dense.py
"""
Couche Dense (Fully Connected)
===============================
Implémente une couche entièrement connectée (dense/fully-connected).
Formule: Y = X @ W + b
"""

import numpy as np
from typing import Optional, Dict, Any
from ..core.layer import Layer


class Dense(Layer):
    """
    Couche Dense (Fully Connected).

    Cette couche implémente l'opération: output = input @ weights + bias
    où @ désigne la multiplication matricielle.

    Attributes:
        units (int): Nombre de neurones dans la couche.
        activation (callable): Fonction d'activation (optionnelle).
        use_bias (bool): Si True, ajoute un biais.
        kernel_initializer (str): Méthode d'initialisation des poids.
        bias_initializer (str): Méthode d'initialisation du biais.
    """

    def __init__(self,
                 units: int,
                 activation: Optional[Any] = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'he_normal',
                 bias_initializer: str = 'zeros',
                 name: Optional[str] = None):
        """
        Initialise la couche Dense.

        Args:
            units (int): Nombre de neurones de sortie.
            activation (callable, optional): Fonction d'activation.
            use_bias (bool): Si True, utilise un biais.
            kernel_initializer (str): Méthode d'initialisation des poids.
            bias_initializer (str): Méthode d'initialisation du biais.
            name (str, optional): Nom de la couche.
        """
        super().__init__(name=name, trainable=True)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        # Paramètres
        self.weights = None
        self.bias = None

        # Gradients
        self.d_weights = None
        self.d_bias = None

        # État de la couche
        self.input_shape = None

    @property
    def kernel(self):
        """Alias pour weights (compatibilité)."""
        return self.weights

    @kernel.setter
    def kernel(self, value):
        """Setter pour kernel."""
        self.weights = value

    def build(self, input_shape: tuple) -> None:
        """
        Construit la couche en initialisant les poids.

        Args:
            input_shape (tuple): Forme des données d'entrée (batch_size, input_dim).
        """
        self.input_shape = input_shape
        input_dim = input_shape[-1]

        # Initialisation des poids
        self.weights = self._initialize_weights(input_dim, self.units)

        if self.use_bias:
            self.bias = self._initialize_bias(self.units)
        else:
            self.bias = None

        self.built = True

    def _initialize_weights(self, input_dim: int, output_dim: int) -> np.ndarray:
        """
        Initialise les poids selon la méthode spécifiée.

        Args:
            input_dim (int): Dimension d'entrée.
            output_dim (int): Dimension de sortie.

        Returns:
            np.ndarray: Matrice de poids initialisée.
        """
        if self.kernel_initializer == 'he_normal':
            return np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        elif self.kernel_initializer == 'he_uniform':
            limit = np.sqrt(6.0 / input_dim)
            return np.random.uniform(-limit, limit, (input_dim, output_dim))
        elif self.kernel_initializer == 'xavier_normal':
            return np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / (input_dim + output_dim))
        elif self.kernel_initializer == 'xavier_uniform':
            limit = np.sqrt(6.0 / (input_dim + output_dim))
            return np.random.uniform(-limit, limit, (input_dim, output_dim))
        elif self.kernel_initializer == 'glorot_normal':
            return np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / (input_dim + output_dim))
        elif self.kernel_initializer == 'glorot_uniform':
            limit = np.sqrt(6.0 / (input_dim + output_dim))
            return np.random.uniform(-limit, limit, (input_dim, output_dim))
        elif self.kernel_initializer == 'orthogonal':
            # Initialisation orthogonale
            a = np.random.randn(input_dim, output_dim)
            u, s, vh = np.linalg.svd(a, full_matrices=False)
            return u if u.shape == (input_dim, output_dim) else vh.T
        elif self.kernel_initializer == 'zeros':
            return np.zeros((input_dim, output_dim))
        elif self.kernel_initializer == 'ones':
            return np.ones((input_dim, output_dim))
        else:
            # Par défaut: Glorot normal
            return np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / (input_dim + output_dim))

    def _initialize_bias(self, output_dim: int) -> np.ndarray:
        """
        Initialise le biais selon la méthode spécifiée.

        Args:
            output_dim (int): Dimension de sortie.

        Returns:
            np.ndarray: Vecteur de biais initialisé.
        """
        if self.bias_initializer == 'zeros':
            return np.zeros((1, output_dim))
        elif self.bias_initializer == 'ones':
            return np.ones((1, output_dim))
        elif self.bias_initializer == 'random_normal':
            return np.random.randn(1, output_dim) * 0.01
        else:
            return np.zeros((1, output_dim))

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant de la couche Dense.

        Args:
            inputs (np.ndarray): Données d'entrée de forme (batch_size, input_dim).
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Sortie de forme (batch_size, units).
        """
        if not self.built:
            self.build(inputs.shape)

        self.input = inputs
        output = np.dot(inputs, self.weights)

        if self.use_bias:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        self.output = output
        return output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière de la couche Dense.

        Args:
            grad_output (np.ndarray): Gradient de la perte par rapport à la sortie.
            optimizer (Optimizer, optional): Optimiseur pour la mise à jour des poids.

        Returns:
            np.ndarray: Gradient de la perte par rapport à l'entrée.
        """
        if self.activation is not None:
            grad_output = self.activation.gradient(grad_output, self.output)

        # Gradient des poids: dL/dW = X.T @ dL/dY
        # On ne divise PAS par len(self.input) ici car la perte le fait déjà.
        grad_weights = np.dot(self.input.T, grad_output)

        if self.use_bias:
            grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        else:
            grad_bias = None

        # Gradient d'entrée: dL/dX = dL/dY @ W.T
        grad_input = np.dot(grad_output, self.weights.T)

        # Mise à jour des poids via l'optimiseur
        if optimizer is not None and self.trainable:
            optimizer.update(self, grad_weights, grad_bias)

        # Stocker les gradients
        self.d_weights = grad_weights
        self.d_bias = grad_bias

        return grad_input

    def get_weights(self) -> list:
        """Retourne les poids de la couche."""
        if self.use_bias:
            return [self.weights, self.bias]
        return [self.weights]

    def set_weights(self, weights: list) -> None:
        """Définit les poids de la couche."""
        self.weights = weights[0]
        if self.use_bias and len(weights) > 1:
            self.bias = weights[1]
        self.built = True

    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration de la couche."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': str(self.activation) if self.activation else None,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
        })
        return config

    def __repr__(self) -> str:
        return f"Dense(units={self.units}, activation={self.activation})"
