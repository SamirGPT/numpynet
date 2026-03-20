# NumPyNet

**Une bibliothèque de Deep Learning complète écrite en pur NumPy (CPU uniquement)**

NumPyNet est une implémentation complète de Keras/TensorFlow en pur Python/NumPy.
Aucune dépendance externe (sauf NumPy) - parfait pour comprendre le fonctionnement interne des réseaux de neurones.

---

## Table des Matières

1. [Installation](#installation)
2. [Guide de Démarrage Rapide](#guide-de-démarrage-rapide)
3. [Architecture de la Bibliothèque](#architecture-de-la-bibliothèque)
4. [API Référence](#api-référence)
5. [Exemples](#exemples)
6. [Architecture des Modèles](#architecture-des-modèles)

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/yourusername/numpynet.git
cd numpynet

# Installer (juste NumPy requis)
pip install numpy

# Ou ajouter au path Python
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

**Requirements:**
- Python 3.8+
- NumPy 1.20+

---

## Guide de Démarrage Rapide

### 1. Classification Simple (XOR)

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
    Dense(4, activation=ReLU()),
    Dense(1, activation=Sigmoid())
])

# Compiler et entraîner
model.compile(optimizer=Adam(), loss=MSE())
history = model.fit(X, y, epochs=1000, verbose=1)

# Prédire
predictions = model.predict(X)
```

### 2. Classification Multi-classes

```python
from numpynet import Dense, Dropout, ReLU, Softmax
from numpynet.losses import CategoricalCrossentropy

# Créer un réseau pour classification
model = Sequential([
    Dense(128, activation=ReLU()),
    Dropout(0.2),
    Dense(64, activation=ReLU()),
    Dense(10, activation=Softmax())
])

model.compile(
    optimizer='adam',           # ou Adam(learning_rate=0.001)
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entraîner
model.fit(x_train, y_train_onehot, epochs=10, batch_size=32)

# Évaluer
results = model.evaluate(x_test, y_test_onehot)
```

### 3. Régression

```python
from numpynet.losses import MSE

model = Sequential([
    Dense(64, activation=ReLU()),
    Dense(64, activation=ReLU()),
    Dense(1)  # Pas d'activation pour régression
])

model.compile(optimizer='adam', loss=MSE())
model.fit(X_train, y_train, epochs=100, batch_size=32)
predictions = model.predict(X_test)
```

---

## Architecture de la Bibliothèque

```
numpynet/
├── core/           # Classes de base
│   ├── layer.py     # Classe Layer abstraite
│   └── model.py     # Classe Model abstraite
├── models/          # Implémentations de modèles
│   └── sequential.py
├── layers/         # Types de couches
│   ├── dense.py          # Dense (FC)
│   ├── conv2d.py          # Convolution 2D
│   ├── pooling.py         # Max/Average Pooling
│   ├── flatten.py         # Aplatir
│   ├── dropout.py         # Regularization
│   └── batch_normalization.py
├── activations/     # Fonctions d'activation
│   ├── relu.py
│   ├── sigmoid.py
│   ├── tanh.py
│   ├── softmax.py
│   └── ...
├── losses/         # Fonctions de perte
│   ├── mse.py
│   ├── binary_crossentropy.py
│   └── categorical_crossentropy.py
├── optimizers/     # Algorithmes d'optimisation
│   ├── sgd.py
│   ├── adam.py
│   ├── rmsprop.py
│   └── ...
└── examples/        # Exemples d'utilisation
    ├── xor.py
    ├── mnist.py
    └── regression.py
```

---

## API Référence

### Modèles

#### `Sequential(layers=None, name=None)`

Modèle séquentiel - empile des couches les unes après les autres.

```python
from numpynet import Sequential, Dense, ReLU

model = Sequential([
    Dense(128, activation=ReLU()),
    Dense(10, activation=Softmax())
])
```

**Méthodes:**
- `add(layer)` - Ajoute une couche
- `compile(optimizer, loss, metrics)` - Compile le modèle
- `fit(x, y, epochs, batch_size, validation_data)` - Entraîne
- `predict(x)` - Fait des prédictions
- `evaluate(x, y)` - Évalue le modèle
- `summary()` - Affiche l'architecture
- `save_weights(filepath)` / `load_weights(filepath)` - Sauvegarde

---

### Couches

#### `Dense(units, activation=None, use_bias=True)`

Couche entièrement connectée.

```python
# Syntaxe simple
Dense(128)

# Avec activation
Dense(128, activation=ReLU())

# Nommeée
Dense(128, activation=ReLU(), name='hidden1')
```

#### `Conv2D(filters, kernel_size, strides=1, padding=0, activation=None)`

Couche de convolution 2D.

```python
Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=ReLU())
Conv2D(filters=64, kernel_size=(5, 5), strides=2, activation=ReLU())
```

#### `MaxPooling2D(pool_size=2, strides=None)` / `AveragePooling2D`

Couches de pooling.

```python
MaxPooling2D(pool_size=(2, 2))
AveragePooling2D(pool_size=(2, 2))
```

#### `Flatten()`

Aplatit l'entrée pour les couches denses.

```python
Flatten()
```

#### `Dropout(rate)`

Régularisation par désactivation aléatoire.

```python
Dropout(0.2)  # 20% des neurones désactivés
```

#### `BatchNormalization(momentum=0.99, epsilon=1e-3)`

Normalise les activations par batch.

```python
BatchNormalization()
```

---

### Fonctions d'Activation

| Classe | Fonction | Dérivée |
|--------|----------|---------|
| `ReLU()` | max(0, x) | 1 si x > 0, 0 sinon |
| `Sigmoid()` | 1/(1+e^-x) | sigmoid * (1-sigmoid) |
| `Tanh()` | (e^x - e^-x)/(e^x + e^-x) | 1 - tanh² |
| `Softmax()` | e^x / sum(e^x) | Jacobian complexe |
| `LeakyReLU(alpha=0.01)` | x si x>0, alpha*x sinon | 1 ou alpha |
| `ELU(alpha=1.0)` | x si x>0, alpha*(e^x-1) sinon | 1 ou alpha*e^x + alpha |
| `Swish(beta=1.0)` | x * sigmoid(beta*x) | - |

```python
from numpynet import ReLU, Sigmoid, Tanh, Softmax

# Usage
Dense(128, activation=ReLU())
Dense(10, activation=Softmax())
```

---

### Fonctions de Perte

| Classe | Usage | Formule |
|--------|-------|---------|
| `MSE()` | Régression | mean((y - y_pred)²) |
| `BinaryCrossentropy()` | Binaire | -y*log(y_pred) - (1-y)*log(1-y_pred) |
| `CategoricalCrossentropy()` | Multi-classes (one-hot) | -sum(y * log(y_pred)) |
| `SparseCategoricalCrossentropy()` | Multi-classes (int) | -log(y_pred[y]) |

```python
from numpynet.losses import MSE, CategoricalCrossentropy

model.compile(optimizer='adam', loss=MSE())  # Régression
model.compile(optimizer='adam', loss='categorical_crossentropy')  # Par string
```

---

### Optimiseurs

| Classe | Description | Paramètres |
|--------|-------------|-------------|
| `SGD(lr, momentum, nesterov)` | Gradient Descent Stochastique | lr=0.01 |
| `Adam(lr, beta1, beta2)` | Adaptive Moment Estimation | lr=0.001 |
| `RMSprop(lr, rho, momentum)` | Root Mean Square Propagation | lr=0.001 |
| `Adagrad(lr)` | Adaptive Gradients | lr=0.01 |
| `Momentum(lr, momentum)` | Momentum classique | lr=0.01 |
| `AdamW(lr, weight_decay)` | Adam avec weight decay | lr=0.001 |

```python
from numpynet.optimizers import SGD, Adam, AdamW

# Par string (recommandé)
model.compile(optimizer='adam', loss='mse')

# Par instance
model.compile(optimizer=Adam(learning_rate=0.001, beta1=0.9), loss='mse')
```

---

## Exemples

Exécuter les exemples:

```bash
# XOR (classification binaire)
python -m numpynet.examples.xor

# Régression
python -m numpynet.examples.regression

# MNIST (classification)
python -m numpynet.examples.mnist
```

### Exemple Complet: CNN pour Images

```python
from numpynet import Sequential, Dense, Dropout, Flatten
from numpynet.layers import Conv2D, MaxPooling2D
from numpynet import ReLU, Softmax

model = Sequential([
    # Bloc convolutif 1
    Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=ReLU()),
    MaxPooling2D(pool_size=(2, 2)),

    # Bloc convolutif 2
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=ReLU()),
    MaxPooling2D(pool_size=(2, 2)),

    # Classification
    Flatten(),
    Dense(128, activation=ReLU()),
    Dropout(0.5),
    Dense(10, activation=Softmax())
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## Architecture des Modèles

### Forward Pass (Passe Avant)

```
Input → Layer1 → Layer2 → ... → LayerN → Output
         ↓         ↓              ↓
      stocke    stocke          stocke
      input     input           output
```

### Backward Pass (Passe Arrière)

```
Output ← LayerN.backward(dL/dy) ← ... ← Layer2.backward ← Layer1.backward
                          ↓                        ↓
                       met à jour              met à jour
                       les poids               les poids
```

### Boucle d'Entraînement

```
Pour chaque époque:
    Mélanger les données

    Pour chaque batch:
        1. forward(inputs) → predictions
        2. loss(y, predictions) → loss_value
        3. loss.backward() → gradients
        4. model.backward(gradients)
           → Chaque couche calcule dL/dX et met à jour ses poids
        5. optimizer.apply_gradients()
```

---

## FAQ

### Q: Comment ajouter une nouvelle couche?

```python
from numpynet.core.layer import Layer
import numpy as np

class MaCouche(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.weights = None
        self.bias = None

    def build(self, input_shape):
        self.weights = np.random.randn(input_shape[-1], self.units) * 0.1
        self.bias = np.zeros((1, self.units))

    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output

    def backward(self, grad_output, optimizer=None):
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)

        if optimizer:
            optimizer.update(self, grad_weights, grad_bias)

        return grad_input
```

### Q: Comment ajouter une nouvelle activation?

```python
import numpy as np

class MonActivation:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return np.tanh(x)  # Exemple

    def gradient(self, grad_output, output):
        return grad_output * (1 - output**2)  # d/dx tanh(x) = 1 - tanh²(x)
```

### Q: Pourquoi mon modèle ne converge pas?

1. **Learning rate trop grand**: Réduisez le lr
2. **Pas assez d'époques**: Augmentez les epochs
3. **Architecture inadaptée**: Trop petit ou trop grand réseau
4. **Données non normalisées**: Normalisez vos entrées
5. **Vanishing gradients**: Utilisez ReLU ou BatchNorm

---

## License

MIT License - Libre d'utilisation et de modification.

---

**Happy Deep Learning! 🚀**
