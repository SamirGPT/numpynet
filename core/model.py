# numpynet/core/model.py
"""
Classe de base Model
====================
Classe abstraite définissant l'interface commune à tous les modèles
de deep learning dans NumPyNet.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time


class Model:
    """
    Classe de base pour tous les modèles neuronaux.

    Cette classe fournit l'interface commune pour la compilation,
    l'entraînement et l'évaluation des modèles.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialise le modèle de base.

        Args:
            name (str, optional): Nom du modèle.
        """
        self.name = name or self.__class__.__name__
        self.built = False
        self.compiled = False
        self.training = True

        # Attributs de compilation
        self.loss = None
        self.optimizer = None
        self.metrics = []

        # Historique d'entraînement
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
        }

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant du modèle.

        Args:
            inputs (np.ndarray): Données d'entrée.
            training (bool): Indique si le modèle est en mode entraînement.

        Returns:
            np.ndarray: Prédictions du modèle.
        """
        raise NotImplementedError("La méthode forward doit être implémentée par les sous-classes")

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Passe arrière du modèle.

        Args:
            grad_output (np.ndarray): Gradient de la perte.
        """
        raise NotImplementedError("La méthode backward doit être implémentée par les sous-classes")

    def compile(self, optimizer: Any, loss: Any, metrics: Optional[List[str]] = None) -> None:
        """
        Compile le modèle avec un optimiseur et une fonction de perte.

        Args:
            optimizer: Optimiseur à utiliser.
            loss: Fonction de perte à utiliser.
            metrics (List[str], optional): Liste des métriques à suivre.
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []
        self.compiled = True

        # Initialiser les états des optimiseurs si nécessaire
        self._init_optimizer_states()

    def _init_optimizer_states(self) -> None:
        """Initialise les états des optimiseurs pour chaque couche entraînable."""
        pass

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int = 1,
            batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: int = 1,
            callbacks: Optional[List[Any]] = None) -> Dict[str, List[float]]:
        """
        Entraîne le modèle sur les données fournies.

        Args:
            x (np.ndarray): Données d'entrée d'entraînement.
            y (np.ndarray): Labels d'entraînement.
            epochs (int): Nombre d'époques d'entraînement.
            batch_size (int): Taille des batches.
            validation_data (tuple, optional): Données de validation (X_val, y_val).
            verbose (int): Niveau de verbosité (0, 1, ou 2).
            callbacks (list, optional): Liste des callbacks à appeler.

        Returns:
            Dict[str, List[float]]: Historique de l'entraînement.
        """
        if not self.compiled:
            raise RuntimeError("Le modèle doit être compilé avant l'entraînement")

        callbacks = callbacks or []
        self.training = True

        # Préparer les données
        num_samples = len(x)

        # Boucle d'entraînement
        for epoch in range(epochs):
            start_time = time.time()

            # Mélanger les données à chaque époque
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            num_batches = 0

            # Traitement par batches
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                x_batch = x_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]

                # Forward pass
                predictions = self.forward(x_batch, training=True)

                # Calcul de la perte
                loss_value = self.loss(y_batch, predictions)
                epoch_loss += loss_value
                num_batches += 1

                # Backward pass
                grad_loss = self.loss.gradient(y_batch, predictions)
                self.backward(grad_loss)

            # Moyenne de la perte
            epoch_loss /= num_batches
            self.history['loss'].append(epoch_loss)

            # Validation
            if validation_data is not None:
                val_predictions = self.forward(validation_data[0], training=False)
                val_loss = self.loss(validation_data[1], val_predictions)
                self.history['val_loss'].append(val_loss)

            # Métriques
            if 'accuracy' in self.metrics:
                predictions_train = self.forward(x, training=False)
                acc = self._calculate_accuracy(y, predictions_train)
                self.history['accuracy'].append(acc)

                if validation_data is not None:
                    acc_val = self._calculate_accuracy(
                        validation_data[1],
                        self.forward(validation_data[0], training=False)
                    )
                    self.history['val_accuracy'].append(acc_val)

            # Affichage
            if verbose > 0:
                elapsed = time.time() - start_time
                msg = f"Epoch {epoch + 1}/{epochs} - {elapsed:.2f}s - loss: {epoch_loss:.4f}"
                if validation_data is not None:
                    msg += f" - val_loss: {val_loss:.4f}"
                if 'accuracy' in self.metrics:
                    msg += f" - accuracy: {acc:.4f}"
                    if validation_data is not None:
                        msg += f" - val_accuracy: {acc_val:.4f}"
                print(msg)

        return self.history

    def evaluate(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 batch_size: int = 32,
                 verbose: int = 1) -> Dict[str, float]:
        """
        Évalue le modèle sur les données de test.

        Args:
            x (np.ndarray): Données d'entrée.
            y (np.ndarray): Labels.
            batch_size (int): Taille des batches.
            verbose (int): Niveau de verbosité.

        Returns:
            Dict[str, float]: Métriques d'évaluation.
        """
        self.training = False

        num_samples = len(x)
        total_loss = 0.0
        num_batches = 0

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            x_batch = x[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]

            predictions = self.forward(x_batch, training=False)
            loss_value = self.loss(y_batch, predictions)
            total_loss += loss_value
            num_batches += 1

        avg_loss = total_loss / num_batches
        results = {'loss': avg_loss}

        if 'accuracy' in self.metrics:
            predictions = self.forward(x, training=False)
            acc = self._calculate_accuracy(y, predictions)
            results['accuracy'] = acc

        if verbose > 0:
            msg = f"Evaluation - loss: {results['loss']:.4f}"
            if 'accuracy' in self.metrics:
                msg += f" - accuracy: {results['accuracy']:.4f}"
            print(msg)

        return results

    def predict(self,
                x: np.ndarray,
                batch_size: int = 32,
                verbose: int = 0) -> np.ndarray:
        """
        Fait des prédictions sur les données d'entrée.

        Args:
            x (np.ndarray): Données d'entrée.
            batch_size (int): Taille des batches.
            verbose (int): Niveau de verbosité.

        Returns:
            np.ndarray: Prédictions du modèle.
        """
        self.training = False
        num_samples = len(x)
        predictions = []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            x_batch = x[batch_start:batch_end]

            batch_predictions = self.forward(x_batch, training=False)
            predictions.append(batch_predictions)

        return np.vstack(predictions)

    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule l'exactitude (accuracy) du modèle.

        Args:
            y_true (np.ndarray): Vraies étiquettes.
            y_pred (np.ndarray): Prédictions.

        Returns:
            float: Exactitude du modèle.
        """
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # Classification multiclasses
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_true, axis=1)
        else:
            # Classification binaire ou régression
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            y_true_classes = y_true.flatten()

        return np.mean(y_pred_classes == y_true_classes)

    def summary(self) -> None:
        """
        Affiche un résumé de l'architecture du modèle.
        """
        print(f"Model: {self.name}")
        print("=" * 60)

    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration du modèle.

        Returns:
            Dict[str, Any]: Configuration du modèle.
        """
        return {
            'name': self.name,
            'compiled': self.compiled,
            'training': self.training,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
