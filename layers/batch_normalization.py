# numpynet/layers/batch_normalization.py
"""
BatchNormalization
=================
Couche de normalisation par lots qui normalise les activations
et apprend des paramètres de scale et shift.
"""

import numpy as np
from typing import Optional, Dict, Any
from ..core.layer import Layer


class BatchNormalization(Layer):
    """
    Couche Batch Normalization.

    Normalise les activations en utilisant la moyenne et l'écart-type
    des mini-batchs courants. Apprend des paramètres gamma (scale)
    et beta (shift) pour permettre au modèle de désapprendre la normalisation
    si nécessaire.

    Formules:
        - Normalisation: x_norm = (x - mean) / sqrt(var + epsilon)
        - Scale & Shift: y = gamma * x_norm + beta

    Pendant l'inférence:
        - Utilise les moyennes et variances移动 (EMA) accumulées

    Attributes:
        momentum (float): Momentum pour la moyenne mobile.
        epsilon (float): Petit terme pour éviter la division par zéro.
        axis (int or tuple): Axe(s) à normaliser.
        center (bool): Si True, ajoute le biais beta.
        scale (bool): Si True, multiplie par gamma.
    """

    def __init__(self,
                 momentum: float = 0.99,
                 epsilon: float = 1e-3,
                 axis: int = -1,
                 center: bool = True,
                 scale: bool = True,
                 name: Optional[str] = None):
        """
        Initialise la couche BatchNormalization.

        Args:
            momentum (float): Momentum pour la moyenne mobile (EMA).
            epsilon (float): Petit terme numérique pour la stabilité.
            axis (int or tuple): Axe de normalisation.
            center (bool): Ajouter un biais.
            scale (bool): Multiplier par gamma.
        """
        super().__init__(name=name, trainable=True)
        self.momentum = momentum
        self.epsilon = epsilon
        self.axis = axis
        self.center = center
        self.scale = scale

        # Paramètres entraînables
        self.gamma = None  # Scale
        self.beta = None   # Shift

        # Moyennes mobiles pour l'inférence
        self.moving_mean = None
        self.moving_var = None

        # État
        self.training_mode = True
        self.batch_mean = None
        self.batch_var = None
        self.normalized = None

    def build(self, input_shape):
        """
        Initialise les poids de la couche.

        Args:
            input_shape (tuple): Forme de l'entrée.
        """
        self.built = True

        # Répéter gamma et beta sur les bons axes
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            # Normalisation sur le dernier axe (features)
            shape = (input_shape[-1],)
        elif isinstance(self.axis, int):
            shape = (input_shape[self.axis],)
        else:
            shape = tuple(input_shape[a] for a in self.axis)

        # Initialisation de Xavier/He
        if self.scale:
            self.gamma = np.ones(shape) * 0.1
        else:
            self.gamma = None

        if self.center:
            self.beta = np.zeros(shape)
        else:
            self.beta = None

        # Moyennes mobiles initialisées
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

        return self

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant de la normalisation par lots.

        Args:
            inputs (np.ndarray): Données d'entrée de forme (batch, ...).
            training (bool): Mode entraînement ou inférence.

        Returns:
            np.ndarray: Activations normalisées.
        """
        self.inputs = inputs

        if training:
            self.training_mode = True

            # Calculer la moyenne et variance du batch
            if self.axis == -1 or self.axis == len(inputs.shape) - 1:
                # Réduire sur toutes les dimensions sauf la dernière
                axis = tuple(range(inputs.ndim - 1))
            elif isinstance(self.axis, int):
                axis = tuple(i for i in range(inputs.ndim) if i != self.axis)
            else:
                axis = tuple(i for i in range(inputs.ndim) if i not in self.axis)

            self.batch_mean = np.mean(inputs, axis=axis, keepdims=True)
            self.batch_var = np.var(inputs, axis=axis, keepdims=True)

            # Normaliser
            self.normalized = (inputs - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)

            # Mise à jour des moyennes mobiles
            if self.moving_mean.shape == ():
                # Scalar case
                self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.batch_mean
                self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.batch_var
            else:
                self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.batch_mean
                self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.batch_var

        else:
            self.training_mode = False
            # Utiliser les moyennes mobiles pour l'inférence
            # Reshape pour broadcast correct
            if self.axis == -1 or self.axis == len(inputs.shape) - 1:
                keepdims_shape = tuple(1 if i == inputs.ndim - 1 else s for i, s in enumerate(inputs.shape))
            else:
                keepdims_shape = self._get_keepdims_shape(inputs.shape)

            mean = self.moving_mean.reshape(keepdims_shape)
            var = self.moving_var.reshape(keepdims_shape)

            self.normalized = (inputs - mean) / np.sqrt(var + self.epsilon)

        # Scale et Shift
        if self.gamma is not None:
            gamma = self.gamma.reshape(self._get_keepdims_shape(inputs.shape))
        else:
            gamma = 1

        if self.beta is not None:
            beta = self.beta.reshape(self._get_keepdims_shape(inputs.shape))
        else:
            beta = 0

        self.output = gamma * self.normalized + beta
        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière de la normalisation par lots.

        Args:
            grad_output (np.ndarray): Gradient de la sortie.
            optimizer (Optimizer, optional): Optimiseur pour les poids.

        Returns:
            np.ndarray: Gradient de l'entrée.
        """
        if self.axis == -1 or self.axis == len(self.inputs.shape) - 1:
            axis = tuple(range(self.inputs.ndim - 1))
        elif isinstance(self.axis, int):
            axis = tuple(i for i in range(self.inputs.ndim) if i != self.axis)
        else:
            axis = tuple(i for i in range(self.inputs.ndim) if i not in self.axis)

        # Calculer les gradients de gamma et beta
        if self.gamma is not None:
            grad_gamma = np.sum(grad_output * self.normalized, axis=axis, keepdims=True)
        else:
            grad_gamma = 0

        if self.beta is not None:
            grad_beta = np.sum(grad_output, axis=axis, keepdims=True)
        else:
            grad_beta = 0

        # Gradient par rapport à x_normalized
        grad_normalized = grad_output * self.gamma.reshape(self._get_keepdims_shape(self.inputs.shape))

        # Gradient par rapport à la variance
        grad_var = np.sum(grad_normalized * (self.inputs - self.batch_mean) * (-0.5) *
                         np.power(self.batch_var + self.epsilon, -1.5), axis=axis, keepdims=True)

        # Gradient par rapport à la moyenne
        grad_mean = np.sum(grad_normalized * (-1 / np.sqrt(self.batch_var + self.epsilon)), axis=axis, keepdims=True)
        grad_mean += grad_var * np.mean(-2 * (self.inputs - self.batch_mean), axis=axis, keepdims=True)

        # Gradient final par rapport à l'entrée
        grad_input = grad_normalized / np.sqrt(self.batch_var + self.epsilon)
        grad_input += grad_var * 2 * (self.inputs - self.batch_mean) / self.inputs.shape[0]
        grad_input += grad_mean / self.inputs.shape[0]

        # Mise à jour des poids
        if self.scale and optimizer is not None:
            self.gamma = optimizer.update(self.gamma, grad_gamma.squeeze(), layer_name=self.name + '_gamma')

        if self.center and optimizer is not None:
            self.beta = optimizer.update(self.beta, grad_beta.squeeze(), layer_name=self.name + '_beta')

        return grad_input

    def _get_keepdims_shape(self, input_shape):
        """Retourne la forme pour le broadcast avec keepdims."""
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            return tuple(1 if i == len(input_shape) - 1 else s for i, s in enumerate(input_shape))
        elif isinstance(self.axis, int):
            return tuple(1 if i == self.axis else s for i, s in enumerate(input_shape))
        else:
            return tuple(1 if i in self.axis else s for i, s in enumerate(input_shape))

    def get_weights(self) -> list:
        """Retourne les poids de la couche."""
        weights = []
        if self.gamma is not None:
            weights.append(self.gamma)
        if self.beta is not None:
            weights.append(self.beta)
        return weights

    def set_weights(self, weights: list) -> None:
        """Définit les poids de la couche."""
        idx = 0
        if self.gamma is not None and idx < len(weights):
            self.gamma = weights[idx]
            idx += 1
        if self.beta is not None and idx < len(weights):
            self.beta = weights[idx]

    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'axis': self.axis,
            'center': self.center,
            'scale': self.scale,
        })
        return config

    def __repr__(self) -> str:
        return f"BatchNormalization(momentum={self.momentum}, epsilon={self.epsilon})"


class LayerNormalization(Layer):
    """
    Layer Normalization.

    Normalise les activations sur toutes les caractéristiques pour chaque样本 individuellement.
    Contrairement à BatchNorm, n'utilise pas de statistiques de batch.

    Attributes:
        epsilon (float): Petit terme pour éviter la division par zéro.
        elementwise_affine (bool): Si True, apprend gamma et beta.
    """

    def __init__(self,
                 epsilon: float = 1e-3,
                 elementwise_affine: bool = True,
                 name: Optional[str] = None):
        """
        Initialise la couche LayerNormalization.

        Args:
            epsilon (float): Petit terme numérique.
            elementwise_affine (bool): Apprendre gamma et beta.
        """
        super().__init__(name=name, trainable=True)
        self.epsilon = epsilon
        self.elementwise_affine = elementwise_affine

        self.gamma = None
        self.beta = None
        self.normalized = None

    def build(self, input_shape):
        """Initialise les poids."""
        self.built = True

        if self.elementwise_affine:
            self.gamma = np.ones(input_shape[1:]) * 0.1
            self.beta = np.zeros(input_shape[1:])
        else:
            self.gamma = None
            self.beta = None

        return self

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Passe avant de LayerNorm."""
        self.inputs = inputs

        # Normaliser sur toutes les caractéristiques (axe 1+)
        mean = np.mean(inputs, axis=tuple(range(1, inputs.ndim)), keepdims=True)
        var = np.var(inputs, axis=tuple(range(1, inputs.ndim)), keepdims=True)

        self.normalized = (inputs - mean) / np.sqrt(var + self.epsilon)

        if self.gamma is not None:
            gamma = self.gamma.reshape((1,) + self.gamma.shape)
            beta = self.beta.reshape((1,) + self.beta.shape)
            self.output = gamma * self.normalized + beta
        else:
            self.output = self.normalized

        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """Passe arrière de LayerNorm."""
        # Simplified backward pass
        grad_input = grad_output / np.sqrt(self.epsilon + 1)

        if self.elementwise_affine and optimizer is not None:
            grad_gamma = np.sum(grad_output * self.normalized, axis=0, keepdims=True)
            grad_beta = np.sum(grad_output, axis=0, keepdims=True)
            self.gamma = optimizer.update(self.gamma, grad_gamma.squeeze()[1:] if grad_gamma.ndim > 1 else grad_gamma.squeeze())
            self.beta = optimizer.update(self.beta, grad_beta.squeeze()[1:] if grad_beta.ndim > 1 else grad_beta.squeeze())

        return grad_input

    def get_weights(self) -> list:
        weights = []
        if self.gamma is not None:
            weights.append(self.gamma)
        if self.beta is not None:
            weights.append(self.beta)
        return weights

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'elementwise_affine': self.elementwise_affine,
        })
        return config

    def __repr__(self) -> str:
        return f"LayerNormalization(epsilon={self.epsilon})"


class GroupNormalization(Layer):
    """
    Group Normalization.

    Normalise les activations en divisant les canaux en groupes.
    Indépendant de la taille du batch.

    Attributes:
        groups (int): Nombre de groupes.
        epsilon (float): Petit terme pour éviter la division par zéro.
    """

    def __init__(self,
                 groups: int = 32,
                 epsilon: float = 1e-3,
                 name: Optional[str] = None):
        """
        Initialise la couche GroupNormalization.

        Args:
            groups (int): Nombre de groupes pour la normalisation.
            epsilon (float): Petit terme numérique.
        """
        super().__init__(name=name, trainable=True)
        self.groups = groups
        self.epsilon = epsilon

        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        """Initialise les poids."""
        self.built = True
        channels = input_shape[-1]

        self.gamma = np.ones(channels) * 0.1
        self.beta = np.zeros(channels)

        return self

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Passe avant de GroupNorm."""
        self.inputs = inputs
        batch, height, width, channels = inputs.shape

        # Réorganiser en groupes
        self.groups = min(self.groups, channels)
        group_size = channels // self.groups

        # Reshape pour la normalisation par groupes
        inputs_reshaped = inputs.reshape(batch, height, width, self.groups, group_size)

        # Calculer moyenne et variance par groupe
        mean = np.mean(inputs_reshaped, axis=(1, 2, 4), keepdims=True)
        var = np.var(inputs_reshaped, axis=(1, 2, 4), keepdims=True)

        # Normaliser
        normalized = (inputs_reshaped - mean) / np.sqrt(var + self.epsilon)

        # Reshape back
        normalized = normalized.reshape(batch, height, width, channels)

        # Scale et shift
        self.output = self.gamma * normalized + self.beta

        return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """Passe arrière simplifiée."""
        grad_input = grad_output / np.sqrt(self.epsilon + 1)

        if optimizer is not None:
            grad_gamma = np.sum(grad_output * self.normalized if hasattr(self, 'normalized') else grad_output,
                              axis=(0, 1, 2))
            grad_beta = np.sum(grad_output, axis=(0, 1, 2))
            self.gamma = optimizer.update(self.gamma, grad_gamma)
            self.beta = optimizer.update(self.beta, grad_beta)

        return grad_input

    def get_weights(self) -> list:
        return [self.gamma, self.beta] if self.gamma is not None else []

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'groups': self.groups,
            'epsilon': self.epsilon,
        })
        return config

    def __repr__(self) -> str:
        return f"GroupNormalization(groups={self.groups})"
