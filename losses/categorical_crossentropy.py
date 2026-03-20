# numpynet/losses/categorical_crossentropy.py
"""
Categorical Crossentropy
=========================
Fonction de perte pour la classification multiclasse.
Formule: CE = -(1/n) * Σ Σ[y_true[i,j] * log(y_pred[i,j])]
"""

import numpy as np
from typing import Optional


class CategoricalCrossentropy:
    """
    Categorical Crossentropy (Entropie Croisée Catégorielle).

    Formule: CE = -(1/n) * Σⱼ y_true[i,j] * log(y_pred[i,j])

    Utilisation:
        - Classification multiclasse
        - Nécessite des labels one-hot encodés
        - À utiliser avec une activation Softmax en sortie
    """

    def __init__(self, from_logits: bool = False, name: Optional[str] = None):
        """
        Initialise la fonction de perte Categorical Crossentropy.

        Args:
            from_logits (bool): Si True, les prédictions sont des logits.
            name (str, optional): Nom de la fonction de perte.
        """
        self.name = name or "categorical_crossentropy"
        self.from_logits = from_logits
        self.predictions = None
        self.targets = None
        self.epsilon = 1e-7

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule la perte."""
        return self.forward(y_true, y_pred)

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule la perte Categorical Crossentropy.

        Args:
            y_true (np.ndarray): Vraies étiquettes (one-hot encodées).
            y_pred (np.ndarray): Prédictions (probabilités).

        Returns:
            float: Perte Categorical Crossentropy.
        """
        self.predictions = y_pred
        self.targets = y_true

        # Clip pour éviter log(0)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # Calcul de la crossentropy: -Σ y_true * log(y_pred)
        loss = -np.sum(y_true * np.log(y_pred), axis=-1)

        return np.mean(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcule le gradient de la perte.

        Pour softmax + crossentropy, le gradient se simplifie en:
        dL/dy_pred = y_pred - y_true

        Args:
            y_true (np.ndarray): Vraies étiquettes.
            y_pred (np.ndarray): Prédictions.

        Returns:
            np.ndarray: Gradient de la perte.
        """
        # Clip pour la stabilité
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # Gradient simplifié pour softmax + crossentropy
        # Note: c'est la raison pour laquelle on combine souvent ces deux
        grad = y_pred - y_true

        return grad / len(y_true)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Alias pour la méthode gradient."""
        return self.gradient(y_true, y_pred)

    def get_config(self) -> dict:
        """Retourne la configuration."""
        return {
            'name': self.name,
            'from_logits': self.from_logits,
        }

    def __repr__(self) -> str:
        return f"CategoricalCrossentropy(from_logits={self.from_logits})"


class KLDivergence:
    """
    Kullback-Leibler Divergence.

    Mesure la "distance" entre deux distributions de probabilité.

    Formule: KL = Σ p_true * log(p_true / p_pred)
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or "kl_divergence"
        self.epsilon = 1e-7

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.clip(y_true, self.epsilon, 1)
        y_pred = np.clip(y_pred, self.epsilon, 1)

        loss = np.sum(y_true * np.log(y_true / y_pred), axis=-1)
        return np.mean(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self.epsilon, 1)
        return -y_true / y_pred / len(y_true)

    def __repr__(self) -> str:
        return f"KLDivergence(name='{self.name}')"


class Poisson:
    """
    Perte de Poisson.

    Utilisée pour les problèmes de comptage.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or "poisson"

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        loss = np.mean(y_pred - y_true * np.log(y_pred + 1e-7))
        return loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        grad = 1 - y_true / (y_pred + 1e-7)
        return grad / len(y_true)

    def __repr__(self) -> str:
        return f"Poisson(name='{self.name}')"


class CosineSimilarity:
    """
    Perte de Similarité Cosinus.

    Mesure la similarité cosinus entre deux vecteurs.
    """

    def __init__(self, axis: int = -1, name: Optional[str] = None):
        self.name = name or "cosine_similarity"
        self.axis = axis

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Normaliser
        y_true_norm = y_true / (np.linalg.norm(y_true, axis=self.axis, keepdims=True) + 1e-7)
        y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=self.axis, keepdims=True) + 1e-7)

        # Similarité cosinus
        similarity = np.sum(y_true_norm * y_pred_norm, axis=self.axis)

        # Convertir en perte (1 - similarité)
        return np.mean(1 - similarity)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Gradient simplifié
        return (y_pred - y_true) / len(y_true)

    def __repr__(self) -> str:
        return f"CosineSimilarity(axis={self.axis})"
