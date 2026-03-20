import numpy as np
import sys
import os

# Ajouter le chemin racine au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numpynet import Sequential, Dense, ReLU, Softmax, Flatten
from numpynet.losses import CategoricalCrossentropy
from numpynet.optimizers import Adam

def run_mnist_mock():
    print("Exemple MNIST avec des données synthétiques...")
    
    # Créer des données synthétiques
    num_samples = 1000
    x_train = np.random.randn(num_samples, 28, 28, 1).astype(np.float32)
    y_train = np.zeros((num_samples, 10))
    y_train[np.arange(num_samples), np.random.randint(0, 10, num_samples)] = 1
    
    x_test = np.random.randn(100, 28, 28, 1).astype(np.float32)
    y_test = np.zeros((100, 10))
    y_test[np.arange(100), np.random.randint(0, 10, 100)] = 1
    
    # Créer le modèle
    model = Sequential([
        Flatten(name='flatten'),
        Dense(128, activation=ReLU(), name='dense1'),
        Dense(64, activation=ReLU(), name='dense2'),
        Dense(10, activation=Softmax(), name='output')
    ])
    
    # Compiler
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Entraîner
    print("Début de l'entraînement...")
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1, validation_data=(x_test, y_test))
    
    # Évaluer
    print("\nÉvaluation sur les données de test:")
    model.evaluate(x_test, y_test)

if __name__ == "__main__":
    run_mnist_mock()
