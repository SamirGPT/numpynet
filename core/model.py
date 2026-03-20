# numpynet/core/model.py
"""
Classe de base Model
====================
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time


class Model:
    """
    Classe de base pour tous les modèles neuronaux.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.built = False
        self.compiled = False
        self.training = True

        self.loss = None
        self.optimizer = None
        self.metrics = []

        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
        }

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> None:
        raise NotImplementedError

    def compile(self, optimizer: Any, loss: Any, metrics: Optional[List[str]] = None) -> None:
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []
        self.compiled = True

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int = 1,
            batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: int = 1,
            callbacks: Optional[List[Any]] = None) -> Dict[str, List[float]]:
        
        if not self.compiled:
            raise RuntimeError("Le modèle doit être compilé avant l'entraînement")

        self.training = True
        num_samples = len(x)

        for epoch in range(epochs):
            start_time = time.time()
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            num_batches = 0

            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                x_batch = x_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]

                # Forward pass
                predictions = self.forward(x_batch, training=True)

                # Loss
                loss_value = self.loss(y_batch, predictions)
                epoch_loss += loss_value
                num_batches += 1

                # Backward pass
                grad_loss = self.loss.gradient(y_batch, predictions)
                self.backward(grad_loss)

            epoch_loss /= num_batches
            self.history['loss'].append(epoch_loss)

            # Metrics and Validation
            log_msg = f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}"
            
            if 'accuracy' in self.metrics:
                # We calculate accuracy on a sample of data or the whole set if small
                train_acc = self.evaluate(x, y, batch_size=batch_size, verbose=0)['accuracy']
                self.history['accuracy'].append(train_acc)
                log_msg += f" - accuracy: {train_acc:.4f}"

            if validation_data is not None:
                val_res = self.evaluate(validation_data[0], validation_data[1], batch_size=batch_size, verbose=0)
                self.history['val_loss'].append(val_res['loss'])
                if 'accuracy' in self.metrics:
                    self.history['val_accuracy'].append(val_res['accuracy'])
                    log_msg += f" - val_loss: {val_res['loss']:.4f} - val_accuracy: {val_res['accuracy']:.4f}"
                else:
                    log_msg += f" - val_loss: {val_res['loss']:.4f}"

            if verbose > 0:
                elapsed = time.time() - start_time
                print(f"{log_msg} - {elapsed:.2f}s")

        return self.history

    def evaluate(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 batch_size: int = 32,
                 verbose: int = 1) -> Dict[str, float]:
        self.training = False
        num_samples = len(x)
        total_loss = 0.0
        num_batches = 0
        all_preds = []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            x_batch = x[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]

            preds = self.forward(x_batch, training=False)
            all_preds.append(preds)
            total_loss += self.loss(y_batch, preds)
            num_batches += 1

        avg_loss = total_loss / num_batches
        results = {'loss': avg_loss}

        if 'accuracy' in self.metrics:
            y_pred = np.vstack(all_preds)
            results['accuracy'] = self._calculate_accuracy(y, y_pred)

        if verbose > 0:
            msg = f"Evaluation - loss: {results['loss']:.4f}"
            if 'accuracy' in results:
                msg += f" - accuracy: {results['accuracy']:.4f}"
            print(msg)

        return results

    def predict(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        self.training = False
        num_samples = len(x)
        predictions = []

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            x_batch = x[batch_start:batch_end]
            predictions.append(self.forward(x_batch, training=False))

        return np.vstack(predictions)

    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.shape[-1] > 1:
            # Multi-class
            if y_true.ndim == 1 or y_true.shape[-1] == 1:
                # Sparse labels
                y_true_labels = y_true.flatten()
            else:
                # One-hot labels
                y_true_labels = np.argmax(y_true, axis=-1)
            y_pred_labels = np.argmax(y_pred, axis=-1)
        else:
            # Binary or Regression (threshold at 0.5)
            y_pred_labels = (y_pred > 0.5).astype(int).flatten()
            y_true_labels = (y_true > 0.5).astype(int).flatten()

        return np.mean(y_pred_labels == y_true_labels)

    def summary(self) -> None:
        print(f"Model: {self.name}")
        print("=" * 60)

    def get_config(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'compiled': self.compiled,
        }
