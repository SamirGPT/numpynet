import numpy as np
import sys
import os

# Ajouter le chemin racine au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numpynet import Sequential, Dense, ReLU
from numpynet.losses import MSE
from numpynet.optimizers import Adam

def run_regression():
    print("Exemple de régression simple...")
    
    # Créer des données synthétiques
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = 2 * X + 1 + np.random.normal(0, 0.1, X.shape)
    
    # Créer le modèle
    model = Sequential([
        Dense(64, activation=ReLU(), name='hidden1'),
        Dense(64, activation=ReLU(), name='hidden2'),
        Dense(1, name='output')
    ])
    
    # Compiler
    model.compile(optimizer=Adam(learning_rate=0.01), loss=MSE())
    
    # Entraîner
    print("Début de l'entraînement...")
    model.fit(X, y, epochs=100, batch_size=32, verbose=1)
    
    # Évaluer
    print("\nÉvaluation:")
    model.evaluate(X, y)
    
    # Prédire
    print("\nPrédictions pour quelques valeurs:")
    X_test = np.array([[-0.5], [0.0], [0.5]], dtype=np.float32)
    y_test = 2 * X_test + 1
    predictions = model.predict(X_test)
    for i in range(len(X_test)):
        print(f"Entrée: {X_test[i][0]:.2f}, Cible: {y_test[i][0]:.2f}, Prédiction: {predictions[i][0]:.4f}")

if __name__ == "__main__":
    run_regression()
