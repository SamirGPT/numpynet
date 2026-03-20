# numpynet/losses/sparse_categorical_crossentropy.py
"""
Sparse Categorical Crossentropy
================================
Fonction de perte pour la classification multiclasse avec labels entiers.
"""

import numpy as np
from typing import Optional


class SparseCategoricalCrossentropy:
    """
    Sparse Categorical Crossentropy.

    Identique à CategoricalCrossentropy mais accepte des labels entiers
    au lieu de labels one-hot encodés.

    Formule: CE = -(1/n) * log(y_pred[class_true])

    Utilisation:
        - Classification multiclasse
        - Labels sous forme d'entiers (0, 1, 2, ..., n_classes-1)
    """

    def __init__(self, from_logits: bool = False, name: Optional[str] = None):
        """
        Initialise la fonction de perte Sparse Categorical Crossentropy.

        Args:
            from_logits (bool): Si True, les prédictions sont des logits.
            name (str, optional): Nom de la fonction de perte.
        """
        self.name = name or "sparse_categorical_crossentropy"
        self.from_logits = from_logits
        self.predictions = None
        self.targets = None
        self.epsilon = 1e-7

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule la perte."""
        return self.forward(y_true, y_pred)

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule la perte Sparse Categorical Crossentropy.

        Args:
            y_true (np.ndarray): Vraies étiquettes (entiers).
            y_pred (np.ndarray): Prédictions (probabilités ou logits).

        Returns:
            float: Perte Sparse Categorical Crossentropy.
        """
        self.predictions = y_pred
        self.targets = y_true

        # Convertir les labels entiers en indices
        # y_true peut avoir la forme (n,) ou (n, 1)
        if y_true.ndim > 1:
            y_true = y_true.flatten()

        # Clip pour éviter log(0)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # Récupérer les probabilités des classes vraies
        # Pour chaque échantillon, prendre la probabilité de la classe vraie
        correct_predictions = y_pred[np.arange(len(y_true)), y_true]

        # Calcul de la crossentropy: -log(probabilité de la classe vraie)
        loss = -np.log(correct_predictions)

        return np.mean(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcule le gradient de la perte.

        Pour softmax + crossentropy:
        dL/dy_pred[i, class] = y_pred[i, class] - 1 si class == true_class
                             = y_pred[i, class] sinon

        Args:
            y_true (np.ndarray): Vraies étiquettes (entiers).
            y_pred (np.ndarray): Prédictions.

        Returns:
            np.ndarray: Gradient de la perte.
        """
        # Clip pour la stabilité
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # Convertir les labels
        if y_true.ndim > 1:
            y_true = y_true.flatten()

        # Créer une matrice de gradients
        batch_size = len(y_true)
        num_classes = y_pred.shape[-1]
        grad = np.copy(y_pred)

        # Mettre -1 pour la classe vraie (car y_pred - y_true où y_true = 1 pour la vraie classe)
        for i in range(batch_size):
            true_class = y_true[i]
            grad[i, true_class] -= 1

        return grad / batch_size

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
        return f"SparseCategoricalCrossentropy(from_logits={self.from_logits})"


class Hinge:
    """
    Hinge Loss (SVM Loss).

    Utilisée pour la classification binaire et multiclasse (SVM).
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or "hinge"

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Pour classification binaire
        if y_pred.ndim == 1 or y_pred.shape[-1] == 1:
            loss = np.mean(np.maximum(0, 1 - y_true * y_pred))
        else:
            # Pour multiclasse
            num_classes = y_pred.shape[-1]
            y_true_onehot = self._to_onehot(y_true, num_classes)
            correct_scores = np.sum(y_true_onehot * y_pred, axis=-1)
            margins = y_pred - correct_scores[:, np.newaxis] + 1
            loss = np.mean(np.maximum(0, margins))

        return loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        num_classes = y_pred.shape[-1]
        y_true_onehot = self._to_onehot(y_true, num_classes)

        # Gradient simplifié
        grad = np.where(y_pred - y_true_onehot + 1 > 0, y_pred, 0)
        grad = grad / len(y_true)

        return grad

    def _to_onehot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        """Convertit les labels en one-hot."""
        onehot = np.zeros((len(y), num_classes))
        onehot[np.arange(len(y)), y.astype(int)] = 1
        return onehot

    def __repr__(self) -> str:
        return f"Hinge(name='{self.name}')"


class SquaredHinge:
    """
    Squared Hinge Loss.

    Variante de Hinge avec une fonction au carré.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or "squared_hinge"

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim == 1 or y_pred.shape[-1] == 1:
            loss = np.mean(np.maximum(0, 1 - y_true * y_pred) ** 2)
        else:
            num_classes = y_pred.shape[-1]
            y_true_onehot = self._to_onehot(y_true, num_classes)
            correct_scores = np.sum(y_true_onehot * y_pred, axis=-1)
            margins = y_pred - correct_scores[:, np.newaxis] + 1
            loss = np.mean(np.maximum(0, margins) ** 2)

        return loss

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        num_classes = y_pred.shape[-1]
        y_true_onehot = self._to_onehot(y_true, num_classes)

        margins = y_pred - y_true_onehot + 1
        grad = 2 * np.maximum(0, margins) * np.where(margins > 0, 1, 0)

        return grad / len(y_true)

    def _to_onehot(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        onehot = np.zeros((len(y), num_classes))
        onehot[np.arange(len(y)), y.astype(int)] = 1
        return onehot

    def __repr__(self) -> str:
        return f"SquaredHinge(name='{self.name}')"
