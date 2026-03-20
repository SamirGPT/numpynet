# NumPyNet

**Une bibliothèque de Deep Learning complète et éducative, implémentée en pur NumPy.**

NumPyNet est une bibliothèque de Deep Learning conçue pour la clarté et la compréhension des mécanismes internes des réseaux de neurones. Entièrement écrite en Python et NumPy, elle ne dépend d'aucune autre bibliothèque de calcul numérique, ce qui en fait un outil idéal pour l'apprentissage et l'expérimentation. Inspirée par des frameworks comme Keras, NumPyNet offre une API intuitive pour construire, entraîner et évaluer des modèles de réseaux de neurones.

## Fonctionnalités Clés

*   **Implémentation pure NumPy** : Comprenez chaque opération sans abstraction complexe.
*   **Architecture Modulaire** : Facile à étendre avec de nouvelles couches, fonctions d'activation, pertes et optimiseurs.
*   **API Keras-like** : Construisez des modèles séquentiels rapidement et intuitivement.
*   **Large éventail de composants** :
    *   **Couches** : `Dense`, `Conv2D`, `DepthwiseConv2D`, `Flatten`, `Reshape`, `Permute`, `RepeatVector`, `Dropout`, `SpatialDropout2D`, `AlphaDropout`, `BatchNormalization`, `LayerNormalization`, `MaxPooling2D`, `AveragePooling2D`.
    *   **Activations** : `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyReLU`, `ELU`, `Swish`.
    *   **Pertes** : `MSE`, `MAE`, `RMSE`, `HuberLoss`, `LogCosh`, `BinaryCrossentropy`, `CategoricalCrossentropy`, `SparseCategoricalCrossentropy`, `KLDivergence`, `Poisson`, `CosineSimilarity`.
    *   **Optimiseurs** : `SGD`, `Adam`, `Adamax`, `Nadam`, `RMSprop`, `Adagrad`, `Momentum`, `AdamW`.
*   **Outils d'évaluation** : Fonctions `compile`, `fit`, `evaluate`, `predict` et `summary` pour un workflow complet.

## Installation

Pour installer NumPyNet, clonez le dépôt et installez les dépendances (uniquement NumPy) :

```bash
git clone https://github.com/SamirGPT/numpynet.git
cd numpynet
pip install -e .
```

**Prérequis :**
*   Python 3.8+
*   NumPy 1.20+

## Guide de Démarrage Rapide

### 1. Classification Binaire (Problème XOR)

```python
import numpy as np
from numpynet import Sequential, Dense, ReLU, Sigmoid
from numpynet.losses import MSE
from numpynet.optimizers import Adam

# Données XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Créer le modèle
model = Sequential([
    Dense(8, activation=ReLU(), name=\'hidden1\'),
    Dense(1, activation=Sigmoid(), name=\'output\')
])

# Compiler et entraîner
model.compile(optimizer=Adam(learning_rate=0.01), loss=MSE(), metrics=[\'accuracy\'])
model.fit(X, y, epochs=500, batch_size=4, verbose=0)

# Évaluer et prédire
print("\nÉvaluation:", model.evaluate(X, y))
print("Prédictions:", model.predict(X))
```

### 2. Classification Multi-classes (MNIST simplifié)

```python
import numpy as np
from numpynet import Sequential, Dense, ReLU, Softmax, Flatten
from numpynet.losses import CategoricalCrossentropy
from numpynet.optimizers import Adam

# Données synthétiques (remplacez par de vraies données MNIST)
x_train = np.random.randn(1000, 28, 28, 1).astype(np.float32)
y_train = np.zeros((1000, 10))
y_train[np.arange(1000), np.random.randint(0, 10, 1000)] = 1

x_test = np.random.randn(100, 28, 28, 1).astype(np.float32)
y_test = np.zeros((100, 10))
y_test[np.arange(100), np.random.randint(0, 10, 100)] = 1

model = Sequential([
    Flatten(name=\'flatten\'),
    Dense(128, activation=ReLU(), name=\'dense1\'),
    Dense(64, activation=ReLU(), name=\'dense2\'),
    Dense(10, activation=Softmax(), name=\'output\')
])

model.compile(optimizer=\'adam\', loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)
```

### 3. Régression Simple

```python
import numpy as np
from numpynet import Sequential, Dense, ReLU
from numpynet.losses import MSE
from numpynet.optimizers import Adam

X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 0.1, X.shape)

model = Sequential([
    Dense(64, activation=ReLU(), name=\'hidden1\'),
    Dense(64, activation=ReLU(), name=\'hidden2\'),
    Dense(1, name=\'output\') # Pas d\'activation pour régression
])

model.compile(optimizer=Adam(learning_rate=0.01), loss=MSE())
model.fit(X, y, epochs=100, batch_size=32, verbose=0)

print("\nÉvaluation:", model.evaluate(X, y))
print("Prédictions pour X=[0.0]:", model.predict(np.array([[0.0]])))
```

## Architecture de la Bibliothèque

```
numpynet/
├── core/           # Classes de base (Layer, Model)
├── models/         # Implémentations de modèles (Sequential)
├── layers/         # Types de couches (Dense, Conv2D, Pooling, etc.)
├── activations/    # Fonctions d\'activation (ReLU, Sigmoid, Softmax, etc.)
├── losses/         # Fonctions de perte (MSE, Crossentropy, etc.)
├── optimizers/     # Algorithmes d\'optimisation (Adam, SGD, etc.)
├── examples/       # Exemples d\'utilisation
├── __init__.py     # Initialisation du package et exports
└── setup.py        # Fichier d\'installation (pip install -e .)
```

## API Référence

### Modèles

#### `Sequential(layers=None, name=None)`

Un modèle séquentiel est une pile linéaire de couches. Vous pouvez construire un modèle en passant une liste d'instances de couches au constructeur, ou en ajoutant des couches une par une avec la méthode `add()`.

**Méthodes Clés :**
*   `add(layer)` : Ajoute une couche au modèle.
*   `compile(optimizer, loss, metrics)` : Configure le processus d'entraînement du modèle.
*   `fit(x, y, epochs, batch_size, validation_data, verbose)` : Entraîne le modèle pour un nombre donné d'époques.
*   `evaluate(x, y, batch_size, verbose)` : Évalue la performance du modèle sur des données de test.
*   `predict(x, batch_size)` : Génère des prédictions pour les échantillons d'entrée.
*   `summary()` : Affiche un résumé textuel de l'architecture du modèle.
*   `save_weights(filepath)` / `load_weights(filepath)` : Sauvegarde et charge les poids du modèle.

### Couches

| Classe | Description | Paramètres Clés |
|---|---|---|
| `Dense` | Couche entièrement connectée. | `units`, `activation`, `use_bias` |
| `Conv2D` | Couche de convolution 2D. | `filters`, `kernel_size`, `strides`, `padding`, `activation` |
| `DepthwiseConv2D` | Convolution par canal. | `kernel_size`, `strides`, `padding`, `depth_multiplier` |
| `MaxPooling2D` | Pooling maximal 2D. | `pool_size`, `strides`, `padding` |
| `AveragePooling2D` | Pooling moyen 2D. | `pool_size`, `strides`, `padding` |
| `Flatten` | Aplatit l'entrée en 1D. | - |
| `Reshape` | Remet en forme l'entrée. | `target_shape` |
| `Permute` | Permute les dimensions de l'entrée. | `perm` |
| `RepeatVector` | Répète l'entrée `n` fois. | `n` |
| `Dropout` | Applique le Dropout pour la régularisation. | `rate` |
| `SpatialDropout2D` | Dropout spatial pour les données 2D. | `rate` |
| `AlphaDropout` | Dropout qui maintient la moyenne et la variance. | `rate` |
| `BatchNormalization` | Normalisation par lots. | `momentum`, `epsilon`, `axis` |
| `LayerNormalization` | Normalisation par couche. | `epsilon` |

### Fonctions d'Activation

| Classe | Description | Formule | Dérivée |
|---|---|---|---|
| `ReLU` | Rectified Linear Unit | `max(0, x)` | `1 si x > 0, 0 sinon` |
| `Sigmoid` | Fonction sigmoïde | `1 / (1 + e^-x)` | `sigmoid * (1 - sigmoid)` |
| `Tanh` | Tangente hyperbolique | `(e^x - e^-x) / (e^x + e^-x)` | `1 - tanh²` |
| `Softmax` | Normalisation exponentielle | `e^x / sum(e^x)` | _Jacobien complexe_ |
| `LeakyReLU` | Leaky Rectified Linear Unit | `x si x>0, alpha*x sinon` | `1 ou alpha` |
| `ELU` | Exponential Linear Unit | `x si x>0, alpha*(e^x-1) sinon` | `1 ou alpha*e^x + alpha` |
| `Swish` | Swish activation | `x * sigmoid(beta*x)` | _Complexe_ |

### Fonctions de Perte

| Classe | Usage Principal | Formule |
|---|---|---|
| `MSE` | Régression | `mean((y_true - y_pred)²) ` |
| `MAE` | Régression (robuste aux outliers) | `mean(|y_true - y_pred|)` |
| `RMSE` | Régression | `sqrt(mean((y_true - y_pred)²))` |
| `HuberLoss` | Régression (mixte MSE/MAE) | _Voir implémentation_ |
| `LogCosh` | Régression (lisse, robuste) | `mean(log(cosh(y_true - y_pred)))` |
| `BinaryCrossentropy` | Classification binaire | `-y*log(y_pred) - (1-y)*log(1-y_pred)` |
| `CategoricalCrossentropy` | Classification multi-classes (one-hot) | `-sum(y_true * log(y_pred))` |
| `SparseCategoricalCrossentropy` | Classification multi-classes (entiers) | `-log(y_pred[y_true])` |
| `KLDivergence` | Mesure de distance entre distributions | `sum(p_true * log(p_true / p_pred))` |
| `Poisson` | Problèmes de comptage | `y_pred - y_true * log(y_pred)` |
| `CosineSimilarity` | Mesure de similarité | `1 - cos(y_true, y_pred)` |

### Optimiseurs

| Classe | Description | Paramètres Clés |
|---|---|---|
| `SGD` | Gradient Descent Stochastique | `learning_rate`, `momentum`, `nesterov` |
| `Adam` | Adaptive Moment Estimation | `learning_rate`, `beta1`, `beta2`, `epsilon` |
| `Adamax` | Variante d'Adam avec norme L-infini | `learning_rate`, `beta1`, `beta2`, `epsilon` |
| `Nadam` | Adam avec Nesterov momentum | `learning_rate`, `beta1`, `beta2`, `epsilon` |
| `RMSprop` | Root Mean Square Propagation | `learning_rate`, `rho`, `epsilon` |
| `Adagrad` | Adaptive Gradients | `learning_rate`, `epsilon` |
| `Momentum` | SGD avec Momentum | `learning_rate`, `momentum` |
| `AdamW` | Adam avec Weight Decay découplé | `learning_rate`, `weight_decay` |

## Exemples

Vous pouvez exécuter les exemples fournis dans le dossier `examples/` pour voir NumPyNet en action :

```bash
python -m numpynet.examples.xor
python -m numpynet.examples.mnist_simple
python -m numpynet.examples.regression
```

## Contribuer

NumPyNet est un projet open-source et les contributions sont les bienvenues ! Si vous souhaitez améliorer la bibliothèque, corriger des bugs ou ajouter de nouvelles fonctionnalités, n'hésitez pas à soumettre une Pull Request.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

**Happy Deep Learning! 🚀**
