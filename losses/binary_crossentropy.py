# numpynet/losses/binary_crossentropy.py
"""
Binary Crossentropy
===================
Fonction de perte pour la classification binaire.
Formule: BCE = -(1/n) * Σ[y_true * log(y_pred) + (1-y_true) * log(1-y_pred)]
"""

import numpy as np
from typing import Optional


class BinaryCrossentropy:
    """
    Binary Crossentropy (Entropie Croisée Binaire).

    Formule: BCE = -(1/n) * Σ[y_true * log(y_pred) + (1-y_true) * log(1-y_pred)]

    Utilisation:
        - Classification binaire
        - Doit être utilisée avec une activation Sigmoid en sortie
    """

    def __init__(self, from_logits: bool = False, name: Optional[str] = None):
        """
        Initialise la fonction de perte Binary Crossentropy.

        Args:
            from_logits (bool): Si True, les prédictions sont des logits (non normalisés).
            name (str, optional): Nom de la fonction de perte.
        """
        self.name = name or "binary_crossentropy"
        self.from_logits = from_logits
        self.predictions = None
        self.targets = None
        self.epsilon = 1e-7  # Pour éviter log(0)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule la perte."""
        return self.forward(y_true, y_pred)

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule la perte Binary Crossentropy.

        Args:
            y_true (np.ndarray): Vraies étiquettes (0 ou 1).
            y_pred (np.ndarray): Prédictions (probabilités entre 0 et 1).

        Returns:
            float: Perte Binary Crossentropy.
        """
        self.predictions = y_pred
        self.targets = y_true

        # Convertir logits en probabilités si nécessaire
        if self.from_logits:
            y_pred = self._sigmoid(y_pred)

        # Clip pour éviter log(0)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # Calcul de la crossentropy
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        return np.mean(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcule le gradient de la perte.

        Formule: dL/dy_pred = (y_pred - y_true) / (y_pred * (1 - y_pred))

        Args:
            y_true (np.ndarray): Vraies étiquettes.
            y_pred (np.ndarray): Prédictions.

        Returns:
            np.ndarray: Gradient de la perte.
        """
        if self.from_logits:
            y_pred = self._sigmoid(y_pred)

        # Clip pour la stabilité numérique
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # Gradient: (y_pred - y_true) / (y_pred * (1 - y_pred))
        # Simplifié pour la classification binaire avec sigmoid
        grad = (y_pred - y_true) / (y_pred * (1 - y_pred) + self.epsilon)

        return grad / len(y_true)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Alias pour la méthode gradient."""
        return self.gradient(y_true, y_pred)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Applique la fonction sigmoid."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def get_config(self) -> dict:
        """Retourne la configuration."""
        return {
            'name': self.name,
            'from_logits': self.from_logits,
        }

    def __repr__(self) -> str:
        return f"BinaryCrossentropy(from_logits={self.from_logits})"


class BinaryFocalLoss:
    """
    Binary Focal Loss.

    Variante de Binary Crossentropy qui pénalise les exemples
    correctement classifies avec une haute confiance.

    Formule: FL = -α * (1-p)^γ * log(p)

    Utile pour les problèmes de classification déséquilibrés.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, name: Optional[str] = None):
        self.name = name or "binary_focal_loss"
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1e-7

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # Calcul du focal loss
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma

        loss = -focal_weight * np.log(p_t)
        return np.mean(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma

        # Gradient simplifié
        grad = focal_weight * (p_t - y_true) / p_t

        return grad / len(y_true)

    def __repr__(self) -> str:
        return f"BinaryFocalLoss(alpha={self.alpha}, gamma={self.gamma})"
