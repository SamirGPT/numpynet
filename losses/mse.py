# numpynet/losses/mse.py
"""
MSE (Mean Squared Error)
========================
Fonction de perte utilisée principalement pour la régression.
Formule: MSE = (1/n) * Σ(y_true - y_pred)²
"""

import numpy as np
from typing import Optional, Any


class MSE:
    """
    Mean Squared Error (Erreur Quadratique Moyenne).

    Formule: MSE = (1/n) * Σ(y_true - y_pred)²

    Utilisation:
        - Régression
        - Problèmes où les grandes erreurs sont pénalisées davantage
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialise la fonction de perte MSE.

        Args:
            name (str, optional): Nom de la fonction de perte.
        """
        self.name = name or "mse"
        self.predictions = None
        self.targets = None

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule la perte MSE.

        Args:
            y_true (np.ndarray): Vraies valeurs.
            y_pred (np.ndarray): Prédictions du modèle.

        Returns:
            float: Perte MSE.
        """
        return self.forward(y_true, y_pred)

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Passe avant - Calcule la perte.

        Args:
            y_true (np.ndarray): Vraies valeurs.
            y_pred (np.ndarray): Prédictions du modèle.

        Returns:
            float: Perte MSE.
        """
        self.predictions = y_pred
        self.targets = y_true
        return np.mean(np.power(y_true - y_pred, 2))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcule le gradient de la perte par rapport aux prédictions.

        Formule: dL/dy_pred = (2/n) * (y_pred - y_true)

        Args:
            y_true (np.ndarray): Vraies valeurs.
            y_pred (np.ndarray): Prédictions du modèle.

        Returns:
            np.ndarray: Gradient de la perte.
        """
        n = len(y_true)
        return 2 * (y_pred - y_true) / n

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Alias pour la méthode gradient.

        Args:
            y_true (np.ndarray): Vraies valeurs.
            y_pred (np.ndarray): Prédictions du modèle.

        Returns:
            np.ndarray: Gradient de la perte.
        """
        return self.gradient(y_true, y_pred)

    def get_config(self) -> dict:
        """Retourne la configuration de la fonction de perte."""
        return {'name': self.name}

    def __repr__(self) -> str:
        return f"MSE(name='{self.name}')"


class MAE:
    """
    Mean Absolute Error (Erreur Absolue Moyenne).

    Formule: MAE = (1/n) * Σ|y_true - y_pred|

    Utilisation:
        - Régression
        - Plus robuste aux outliers que MSE
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or "mae"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.forward(y_true, y_pred)

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = len(y_true)
        return np.sign(y_pred - y_true) / n

    def __repr__(self) -> str:
        return f"MAE(name='{self.name}')"


class RMSE:
    """
    Root Mean Squared Error (Racine de l'Erreur Quadratique Moyenne).

    Formule: RMSE = √MSE
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or "rmse"
        self.mse = MSE()

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(self.mse(y_true, y_pred))

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(self.mse.forward(y_true, y_pred))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        mse_grad = self.mse.gradient(y_true, y_pred)
        return mse_grad / (2 * np.sqrt(self.mse(y_true, y_pred)))

    def __repr__(self) -> str:
        return f"RMSE(name='{self.name}')"


class HuberLoss:
    """
    Huber Loss - Combination de MSE et MAE.

    - MSE pour les petites erreurs (|error| < delta)
    - MAE pour les grandes erreurs (|error| >= delta)

    Plus robuste aux outliers que MSE.
    """

    def __init__(self, delta: float = 1.0, name: Optional[str] = None):
        self.name = name or "huber"
        self.delta = delta

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.forward(y_true, y_pred)

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        error = y_true - y_pred
        abs_error = np.abs(error)

        # MSE pour les petites erreurs
        small_error = np.where(abs_error <= self.delta,
                              0.5 * error ** 2,
                              self.delta * (abs_error - 0.5 * self.delta))

        return np.mean(small_error)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        error = y_true - y_pred

        # Dérivée: error pour |error| < delta, sinon delta * sign(error)
        grad = np.where(np.abs(error) <= self.delta,
                        error,
                        self.delta * np.sign(error))

        return grad / len(y_true)

    def __repr__(self) -> str:
        return f"HuberLoss(delta={self.delta})"


class LogCosh:
    """
    Log-Cosh Loss.

    Formule: L = log(cosh(y_pred - y_true))

    Avantages:
        - Similar à Huber mais deux fois différentiable
        - Plus doux que Huber
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or "logcosh"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.forward(y_true, y_pred)

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        diff = y_true - y_pred
        return np.mean(np.log(np.cosh(diff)))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        diff = y_true - y_pred
        return np.tanh(diff) / len(y_true)

    def __repr__(self) -> str:
        return f"LogCosh(name='{self.name}')"
