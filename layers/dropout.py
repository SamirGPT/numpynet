# numpynet/layers/dropout.py
"""
Dropout
=======
Couche de régularisation qui désactive aléatoirement des neurones pendant l'entraînement.
"""

import numpy as np
from typing import Optional, Dict, Any
from ..core.layer import Layer


class Dropout(Layer):
    """
    Couche Dropout.

    Désactive aléatoirement des neurones pendant l'entraînement pour éviter
    le surapprentissage (overfitting).

    Pendant l'entraînement:
        - Chaque neurone a une probabilité `rate` d'être désactivé
        - Les neurones actifs sont mis à l'échelle par 1/(1-rate)

    Pendant l'inférence:
        - Tous les neurones sont actifs (pas de dropout)
        - Pas de mise à l'échelle nécessaire

    Attributes:
        rate (float): Taux de désactivation (entre 0 et 1).
        noise_shape (tuple): Forme du masque de bruit.
        seed (int): Graine aléatoire pour la reproductibilité.
    """

    def __init__(self,
                 rate: float,
                 noise_shape: Optional[tuple] = None,
                 seed: Optional[int] = None,
                 name: Optional[str] = None):
        """
        Initialise la couche Dropout.

        Args:
            rate (float): Taux de dropout (probabilité de désactivation).
            noise_shape (tuple, optional): Forme du masque de bruit.
            seed (int, optional): Graine aléatoire.
        """
        super().__init__(name=name, trainable=False)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.training = True
        self.mask = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant du dropout.

        Args:
            inputs (np.ndarray): Données d'entrée.
            training (bool): Indique si le modèle est en entraînement.

        Returns:
            np.ndarray: Sortie avec dropout (si entraînement) ou identité (si inférence).
        """
        self.training = training

        if training and self.rate > 0:
            # Générer le masque de bruit
            if self.seed is not None:
                np.random.seed(self.seed)

            # Déterminer la forme du masque
            if self.noise_shape is None:
                noise_shape = inputs.shape
            else:
                noise_shape = self.noise_shape

            # Générer le masque (1 = garder, 0 = dropout)
            self.mask = np.random.binomial(1, 1 - self.rate, noise_shape)

            # Mise à l'échelle pendant l'entraînement
            scale = 1 / (1 - self.rate)
            self.output = inputs * self.mask * scale

            return self.output
        else:
            # Pendant l'inférence, pas de dropout
            self.mask = np.ones_like(inputs)
            self.output = inputs
            return self.output

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière du dropout.

        Le gradient ne passe que par les neurones actifs.

        Args:
            grad_output (np.ndarray): Gradient de la sortie.
            optimizer (Optimizer, optional): Non utilisé.

        Returns:
            np.ndarray: Gradient de l'entrée.
        """
        if self.training and self.rate > 0:
            scale = 1 / (1 - self.rate)
            return grad_output * self.mask * scale
        else:
            return grad_output

    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            'rate': self.rate,
            'noise_shape': self.noise_shape,
            'seed': self.seed,
        })
        return config

    def __repr__(self) -> str:
        return f"Dropout(rate={self.rate})"


class SpatialDropout2D(Dropout):
    """
    Spatial Dropout 2D.

    Désactive des canaux entiers au lieu de neurones individuels.
    Plus efficace pour les couches convolutives.
    """

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.training = training

        if training and self.rate > 0:
            # Shape: (batch, height, width, channels)
            # Dropout sur les canaux (dimension 3)
            if self.seed is not None:
                np.random.seed(self.seed)

            noise_shape = (inputs.shape[0], 1, 1, inputs.shape[3])
            self.mask = np.random.binomial(1, 1 - self.rate, noise_shape)

            scale = 1 / (1 - self.rate)
            self.output = inputs * self.mask * scale

            return self.output
        else:
            self.mask = np.ones((inputs.shape[0], 1, 1, inputs.shape[3]))
            self.output = inputs
            return self.output

    def __repr__(self) -> str:
        return f"SpatialDropout2D(rate={self.rate})"


class AlphaDropout(Dropout):
    """
    Alpha Dropout.

    Variante de dropout qui préserve la moyenne et la variance des activations.
    Utilisé avec SELU.
    """

    def __init__(self, rate: float, name: Optional[str] = None):
        super().__init__(rate=rate, name=name)
        self.alpha = -1.758099340847477  # alpha pour SELU

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.training = training

        if training and self.rate > 0:
            if self.seed is not None:
                np.random.seed(self.seed)

            noise_shape = inputs.shape
            self.mask = np.random.binomial(1, 1 - self.rate, noise_shape)

            # Calculer le alpha' pour maintenir la moyenne
            # E[output] = input
            alpha_prime = -self._get_alpha_prime()

            self.output = np.where(
                self.mask == 0,
                alpha_prime,
                inputs
            )

            return self.output
        else:
            self.output = inputs
            return self.output

    def _get_alpha_prime(self):
        # alpha' pour Alpha Dropout
        a = 1 - self.rate
        alpha_prime = self.alpha * (a + self.alpha * self.rate) / a
        return alpha_prime

    def __repr__(self) -> str:
        return f"AlphaDropout(rate={self.rate})"


class Dropout2D(Dropout):
    """
    Dropout 2D.

    Alias pour SpatialDropout2D dans certains frameworks.
    """
    pass


class Dropout3D(Dropout):
    """
    Dropout 3D.

    Pour les entrées 5D (vidéos, volumes).
    """

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.training = training

        if training and self.rate > 0:
            if self.seed is not None:
                np.random.seed(self.seed)

            # Dropout sur les canaux (depth)
            noise_shape = (inputs.shape[0], 1, 1, 1, inputs.shape[4])
            self.mask = np.random.binomial(1, 1 - self.rate, noise_shape)

            scale = 1 / (1 - self.rate)
            self.output = inputs * self.mask * scale

            return self.output
        else:
            self.output = inputs
            return self.output

    def __repr__(self) -> str:
        return f"Dropout3D(rate={self.rate})"
