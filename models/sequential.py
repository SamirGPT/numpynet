# numpynet/models/sequential.py
"""
Modèle Séquentiel
=================
Classe Sequential pour construire des réseaux de neurones couche par couche.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time
from ..core.model import Model


class Sequential(Model):
    """
    Modèle séquentiel de réseau de neurones.

    Permet de construire un réseau de neurones en empilant des couches
    les unes après les autres. Chaque couche prend la sortie de la précédente
    comme entrée.

    Example:
        >>> model = Sequential([
        ...     Dense(128, activation='relu'),
        ...     Dropout(0.2),
        ...     Dense(10, activation='softmax')
        ... ])
        >>> model.compile(optimizer=Adam(), loss='categorical_crossentropy')
        >>> history = model.fit(x_train, y_train, epochs=10)
    """

    def __init__(self, layers: Optional[List[Any]] = None, name: Optional[str] = None):
        """
        Initialise le modèle séquentiel.

        Args:
            layers (list, optional): Liste des couches à ajouter.
            name (str, optional): Nom du modèle.
        """
        super().__init__(name=name or 'sequential')
        self.layers_list = []
        self.layer_names = set()

        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer: Any) -> None:
        """
        Ajoute une couche au modèle.

        Args:
            layer: Instance de couche à ajouter.

        Raises:
            TypeError: Si layer n'hérite pas de Layer.
        """
        from ..core.layer import Layer

        if not isinstance(layer, Layer):
            raise TypeError(f"L'objet ajouté doit être une couche (Layer), reçu: {type(layer)}")

        # Générer un nom unique si nécessaire
        if layer.name in self.layer_names:
            counter = 1
            base_name = layer.name
            while f"{base_name}_{counter}" in self.layer_names:
                counter += 1
            layer.name = f"{base_name}_{counter}"

        self.layer_names.add(layer.name)
        self.layers_list.append(layer)

        # Marquer comme non built si une nouvelle couche est ajoutée
        self.built = False

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Construit le modèle en initialisant les formes des entrées/sorties.

        Args:
            input_shape (tuple): Forme de l'entrée (sans batch size).
        """
        self.input_shape = input_shape
        output_shape = input_shape

        for layer in self.layers_list:
            # Informer la couche de sa forme d'entrée et l'initialiser
            if hasattr(layer, 'build') and callable(layer.build):
                layer.build(output_shape)
            output_shape = self._get_layer_output_shape(layer, output_shape)
            layer.output_shape = output_shape

        self.built = True

    def _get_layer_output_shape(self, layer, input_shape):
        """Calcule la forme de sortie d'une couche."""
        if hasattr(layer, 'output_shape'):
            return layer.output_shape

        # Normaliser input_shape - utiliser la dernière dimension comme feature
        if isinstance(input_shape, tuple):
            if len(input_shape) == 1:
                in_features = input_shape[0]
            else:
                in_features = input_shape[-1]  # Ignore batch dimension
        else:
            in_features = input_shape

        # Couche Dense - calculer sortie basée sur units
        if layer.__class__.__name__ == 'Dense':
            return (layer.units,)
        # Flatten
        if layer.__class__.__name__ == 'Flatten':
            return (np.prod(input_shape),)
        # Dropout, Activation - passent la forme à travers
        if layer.__class__.__name__ in ['Dropout', 'ReLU', 'Sigmoid', 'Tanh', 'Softmax',
                                         'LeakyReLU', 'ELU', 'Swish', 'BatchNormalization',
                                         'LayerNormalization', 'GroupNormalization']:
            return input_shape
        # Conv2D
        if layer.__class__.__name__ == 'Conv2D':
            if len(input_shape) >= 3:
                h, w = input_shape[-3], input_shape[-2]
            else:
                h, w = input_shape[0], input_shape[0]
            filters = layer.filters
            kh, kw = layer.kernel_size
            sh, sw = layer.strides if isinstance(layer.strides, tuple) else (layer.strides, layer.strides)
            ph, pw = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)

            if isinstance(layer.padding, str) and layer.padding.lower() == 'same':
                out_h = int(np.ceil(h / sh))
                out_w = int(np.ceil(w / sw))
            else:
                out_h = (h - kh + 2 * ph) // sh + 1
                out_w = (w - kw + 2 * pw) // sw + 1

            return (out_h, out_w, filters)

        # Pooling
        if 'Pooling2D' in layer.__class__.__name__:
            if len(input_shape) >= 3:
                h, w = input_shape[-3], input_shape[-2]
            else:
                h, w = input_shape[0], input_shape[0]
            ph, pw = layer.pool_size if isinstance(layer.pool_size, tuple) else (layer.pool_size, layer.pool_size)
            sh, sw = layer.strides if isinstance(layer.strides, tuple) else (layer.pool_size, layer.pool_size)

            out_h = int(np.ceil((h - ph + 1) / sh))
            out_w = int(np.ceil((w - pw + 1) / sw))

            channels = input_shape[-1] if len(input_shape) >= 4 else 1
            return (out_h, out_w, channels)

        return input_shape

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Passe avant à travers toutes les couches.

        Args:
            inputs (np.ndarray): Données d'entrée.
            training (bool): Mode entraînement ou inférence.

        Returns:
            np.ndarray: Prédictions du modèle.
        """
        self.inputs = inputs
        output = inputs

        for layer in self.layers_list:
            if hasattr(layer, 'forward'):
                # Couche avec méthode forward
                if 'training' in layer.forward.__code__.co_varnames:
                    output = layer.forward(output, training=training)
                else:
                    output = layer.forward(output)
            else:
                output = layer(output)

            layer.output = output

        return output

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Passe arrière à travers toutes les couches.

        Args:
            grad_output (np.ndarray): Gradient de la perte.
        """
        grad = grad_output

        # Parcourir les couches en sens inverse
        for layer in reversed(self.layers_list):
            if layer.trainable and hasattr(layer, 'backward'):
                grad = layer.backward(grad, self.optimizer)
            else:
                # Pour les couches sans backward (ex: activation en place)
                if hasattr(layer, 'backward'):
                    grad = layer.backward(grad)

    def _init_optimizer_states(self) -> None:
        """Initialise les états des optimiseurs pour chaque couche."""
        for layer in self.layers_list:
            if layer.trainable and hasattr(layer, 'get_weights'):
                weights = layer.get_weights()
                for i, w in enumerate(weights):
                    if hasattr(self.optimizer, 'init_state'):
                        self.optimizer.init_state(w, layer_name=layer.name, weight_index=i)

    def compile(self,
                optimizer: Any,
                loss: Any,
                metrics: Optional[List[str]] = None,
                loss_weights: Optional[List[float]] = None,
                weighted_metrics: Optional[List[str]] = None) -> None:
        """
        Compile le modèle avec un optimiseur et une fonction de perte.

        Args:
            optimizer: Optimiseur (string ou instance).
            loss: Fonction de perte (string ou instance).
            metrics: Liste des métriques à suivre.
            loss_weights: Poids pour chaque sortie (multi-output).
            weighted_metrics: Métriques pondérées.
        """
        # Import pour la conversion des strings
        from ..optimizers import SGD, Adam, RMSprop, Adagrad, Momentum, AdamW
        from ..losses import MSE, BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy

        # Convertir string en objet si nécessaire
        if isinstance(optimizer, str):
            opt_map = {
                'sgd': SGD(),
                'adam': Adam(),
                'rmsprop': RMSprop(),
                'adagrad': Adagrad(),
                'momentum': Momentum(),
                'adamw': AdamW(),
            }
            optimizer = opt_map.get(optimizer.lower(), Adam())

        if isinstance(loss, str):
            loss_map = {
                'mse': MSE(),
                'mean_squared_error': MSE(),
                'binary_crossentropy': BinaryCrossentropy(),
                'categorical_crossentropy': CategoricalCrossentropy(),
                'sparse_categorical_crossentropy': SparseCategoricalCrossentropy(),
            }
            loss = loss_map.get(loss.lower(), MSE())

        super().compile(optimizer, loss, metrics)

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int = 1,
            batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            validation_split: float = 0.0,
            verbose: int = 1,
            callbacks: Optional[List[Any]] = None,
            shuffle: bool = True) -> Dict[str, List[float]]:
        """
        Entraîne le modèle sur les données fournies.

        Args:
            x (np.ndarray): Données d'entrée.
            y (np.ndarray): Labels.
            epochs (int): Nombre d'époques.
            batch_size (int): Taille des batches.
            validation_data (tuple): (X_val, y_val).
            validation_split (float): Fraction pour validation.
            verbose (int): Niveau de verbosité (0, 1, 2).
            callbacks (list): Callbacks Keras-style.
            shuffle (bool): Mélanger les données.

        Returns:
            Dict: Historique de l'entraînement.
        """
        # Construire le modèle si nécessaire
        if not self.built:
            if x.ndim > 2:
                input_shape = x.shape[1:]
            else:
                input_shape = (x.shape[1],)
            self.build(input_shape)

        # Validation split
        if validation_split > 0 and validation_data is None:
            split_idx = int(len(x) * (1 - validation_split))
            x_val, y_val = x[split_idx:], y[split_idx:]
            x, y = x[:split_idx], y[:split_idx]
            validation_data = (x_val, y_val)

        return super().fit(x, y, epochs, batch_size, validation_data, verbose, callbacks)

    def evaluate(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 batch_size: int = 32,
                 verbose: int = 1) -> Dict[str, float]:
        """
        Évalue le modèle.

        Args:
            x: Données d'entrée.
            y: Labels.
            batch_size: Taille des batches.
            verbose: Niveau de verbosité.

        Returns:
            Dict: Métriques d'évaluation.
        """
        if not self.built:
            if x.ndim > 2:
                input_shape = x.shape[1:]
            else:
                input_shape = (x.shape[1],)
            self.build(input_shape)

        return super().evaluate(x, y, batch_size, verbose)

    def predict(self,
                x: np.ndarray,
                batch_size: int = 32,
                verbose: int = 0) -> np.ndarray:
        """
        Fait des prédictions.

        Args:
            x: Données d'entrée.
            batch_size: Taille des batches.
            verbose: Niveau de verbosité.

        Returns:
            np.ndarray: Prédictions.
        """
        if not self.built:
            if x.ndim > 2:
                input_shape = x.shape[1:]
            else:
                input_shape = (x.shape[1],)
            self.build(input_shape)

        return super().predict(x, batch_size, verbose)

    def summary(self) -> None:
        """
        Affiche un résumé de l'architecture du modèle.
        """
        print(f"{'=' * 70}")
        print(f"Model: {self.name}")
        print(f"{'=' * 70}")

        if hasattr(self, 'input_shape'):
            print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':<10}")
            print(f"{'-' * 65}")
            print(f"{'Input':<30} {str(self.input_shape):<25} {'0':<10}")

            total_params = 0
            trainable_params = 0

            for layer in self.layers_list:
                output_shape = getattr(layer, 'output_shape', '?')
                params = self._count_params(layer)

                if layer.trainable:
                    trainable_params += params
                total_params += params

                print(f"{layer.name + ' (' + layer.__class__.__name__ + ')':<30} "
                      f"{str(output_shape):<25} {params:<10}")

            print(f"{'-' * 65}")
            print(f"Total params: {total_params:,}")
            print(f"Trainable params: {trainable_params:,}")
            print(f"Non-trainable params: {total_params - trainable_params:,}")
            print(f"{'=' * 70}")

    def _count_params(self, layer) -> int:
        """Compte le nombre de paramètres d'une couche."""
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            return sum(w.size for w in weights) if weights else 0
        return 0

    def get_layer(self, index: Optional[int] = None, name: Optional[str] = None):
        """
        Retourne une couche par index ou par nom.

        Args:
            index (int): Index de la couche.
            name (str): Nom de la couche.

        Returns:
            Layer: La couche demandée.
        """
        if index is not None:
            return self.layers_list[index]
        if name is not None:
            for layer in self.layers_list:
                if layer.name == name:
                    return layer
            raise ValueError(f"Couche non trouvée: {name}")
        raise ValueError("Spécifiez index ou name")

    def pop(self) -> Any:
        """
        Retire la dernière couche.

        Returns:
            Layer: La couche retirée.
        """
        if not self.layers_list:
            raise RuntimeError("Aucune couche à retirer")
        layer = self.layers_list.pop()
        self.layer_names.discard(layer.name)
        return layer

    def save_weights(self, filepath: str) -> None:
        """
        Sauvegarde les poids du modèle.

        Args:
            filepath (str): Chemin du fichier.
        """
        weights = []
        for layer in self.layers_list:
            if hasattr(layer, 'get_weights'):
                weights.append(layer.get_weights())

        np.save(filepath, weights)

    def load_weights(self, filepath: str) -> None:
        """
        Charge les poids du modèle.

        Args:
            filepath (str): Chemin du fichier.
        """
        weights = np.load(filepath, allow_pickle=True)
        for i, layer in enumerate(self.layers_list):
            if hasattr(layer, 'set_weights') and i < len(weights):
                layer.set_weights(weights[i])

    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration du modèle."""
        config = super().get_config()
        config.update({
            'layers': [layer.get_config() for layer in self.layers_list],
        })
        return config

    def __repr__(self) -> str:
        layers_repr = '\n'.join([f"  {i}: {layer}" for i, layer in enumerate(self.layers_list)])
        return f"Sequential(\n{layers_repr}\n)"
