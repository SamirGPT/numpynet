# numpynet/core/layer.py
"""
Couche de base (Layer Base Class)
==================================
Cette classe abstraite définit l'interface commune à toutes les couches
de neurones dans NumPyNet. Toute couche personnalisée doit hériter de cette classe.
"""

import numpy as np
from typing import Optional, Dict, Any


class Layer:
    """
    Classe de base pour toutes les couches neuronales.

    Attributes:
        name (str): Nom de la couche pour l'identification.
        trainable (bool): Si True, les poids de la couche sont entraînables.
        input (np.ndarray): Dernière entrée reçue (utilisée pour le backward pass).
        output (np.ndarray): Dernière sortie calculée (utilisée pour le backward pass).
    """

    def __init__(self, name: Optional[str] = None, trainable: bool = True):
        """
        Initialise la couche de base.

        Args:
            name (str, optional): Nom de la couche. Si None, un nom par défaut est généré.
            trainable (bool): Si True, les poids peuvent être mis à jour pendant l'entraînement.
        """
        self.name = name or self.__class__.__name__
        self.trainable = trainable
        self.input = None
        self.output = None
        self.built = False  # Indique si la couche a été initialisée

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Permet d'appeler la couche comme une fonction (syntaxe Keras).

        Args:
            inputs (np.ndarray): Données d'entrée.

        Returns:
            np.ndarray: Sortie de la couche.
        """
        return self.forward(inputs)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Passe avant (Forward Pass) - Calcule la sortie à partir de l'entrée.

        Args:
            inputs (np.ndarray): Tension d'entrée.

        Returns:
            np.ndarray: Tension de sortie.

        Raises:
            NotImplementedError: Doit être implémenté par les sous-classes.
        """
        raise NotImplementedError("La méthode forward doit être implémentée par les sous-classes")

    def backward(self, grad_output: np.ndarray, optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Passe arrière (Backward Pass) - Calcule les gradients.

        Args:
            grad_output (np.ndarray): Gradient de la perte par rapport à la sortie.
            optimizer (Optimizer, optional): Optimiseur pour la mise à jour des poids.

        Returns:
            np.ndarray: Gradient de la perte par rapport à l'entrée.

        Raises:
            NotImplementedError: Doit être implémenté par les sous-classes.
        """
        raise NotImplementedError("La méthode backward doit être implémentée par les sous-classes")

    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration de la couche pour la sérialisation.

        Returns:
            Dict[str, Any]: Configuration de la couche.
        """
        return {
            'name': self.name,
            'trainable': self.trainable,
            'built': self.built,
        }

    def get_weights(self) -> list:
        """
        Retourne les poids de la couche.

        Returns:
            list: Liste des poids (tableaux numpy).
        """
        return []

    def set_weights(self, weights: list) -> None:
        """
        Définit les poids de la couche.

        Args:
            weights (list): Liste des poids à définir.
        """
        # À implémenter par les sous-classes avec des poids
        pass

    def summary(self) -> None:
        """
        Affiche un résumé de la couche.
        """
        print(f"{self.name}: {self.__class__.__name__}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', trainable={self.trainable})"
