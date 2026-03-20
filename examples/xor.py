import numpy as np
import sys
import os

# Ajouter le chemin racine au PYTHONPATH pour importer numpynet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numpynet import Sequential, Dense, ReLU, Sigmoid
from numpynet.losses import MSE
from numpynet.optimizers import Adam

def run_xor():
    print("Entraînement du modèle XOR...")
    
    # Données XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # Créer le modèle
    model = Sequential([
        Dense(8, activation=ReLU(), name='hidden1'),
        Dense(1, activation=Sigmoid(), name='output')
    ])
    
    # Compiler
    model.compile(optimizer=Adam(learning_rate=0.01), loss=MSE(), metrics=['accuracy'])
    
    # Entraîner
    print("Début de l'entraînement...")
    model.fit(X, y, epochs=500, batch_size=4, verbose=1)
    
    # Évaluer
    print("\nÉvaluation:")
    results = model.evaluate(X, y)
    
    # Prédire
    print("\nPrédictions:")
    predictions = model.predict(X)
    for i in range(len(X)):
        print(f"Entrée: {X[i]}, Cible: {y[i]}, Prédiction: {predictions[i][0]:.4f}")

if __name__ == "__main__":
    run_xor()
