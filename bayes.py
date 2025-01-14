import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("fish_data.csv")

# Separar características (X) y etiquetas (y)
X = df[['length', 'weight', 'w_l_ratio']].values
y = df['species'].values

# Implementación manual de Naive Bayes (Gaussian)
class NaiveBayesManual:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.mean[cls] = np.mean(X_cls, axis=0)
            self.var[cls] = np.var(X_cls, axis=0)
            self.priors[cls] = len(X_cls) / len(X)

    def _gaussian_probability(self, x, mean, var):
        coeff = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coeff * exponent

    def predict_proba(self, X):
        probabilities = []
        for x in X:
            cls_probs = {}
            for cls in self.classes:
                likelihood = np.prod(self._gaussian_probability(x, self.mean[cls], self.var[cls]))
                cls_probs[cls] = self.priors[cls] * likelihood
            total_prob = sum(cls_probs.values())
            probabilities.append({cls: cls_probs[cls] / total_prob for cls in cls_probs})
        return probabilities

    def predict(self, X):
        probas = self.predict_proba(X)
        predictions = [max(p, key=p.get) for p in probas]
        return predictions

# Convertir etiquetas a valores numéricos para cálculos
y_encoded = np.array([np.where(np.unique(y) == label)[0][0] for label in y])

# Crear y entrenar el modelo
modelo = NaiveBayesManual()
modelo.fit(X, y)

# Función para realizar predicciones
def predecir_especie(length, weight, w_l_ratio):
    """
    Realiza una predicción de la especie con base en las características dadas.
    Args:
        length (float): Longitud del pez.
        weight (float): Peso del pez.
        w_l_ratio (float): Relación peso-longitud del pez.
    Returns:
        str: Predicción de la especie.
    """
    nuevo_pez = np.array([[length, weight, w_l_ratio]])
    prediccion = modelo.predict(nuevo_pez)
    probabilidades = modelo.predict_proba(nuevo_pez)

    print(f"\nPredicción: {prediccion[0]}")
    print("Probabilidades:")
    for clase, prob in probabilidades[0].items():
        print(f"  Especie: {clase}, Probabilidad: {prob:.2f}")
    return prediccion[0]

# Graficar las características con colores por especie
sns.scatterplot(data=df, x='length', y='weight', hue='species', palette='viridis')
plt.xlabel("Length (cm)")
plt.ylabel("Weight (g)")
plt.title("Length vs Weight from Fish Species")
plt.show()

# Visualizar probabilidades (simulando PCA con las dos primeras características)
probas = modelo.predict_proba(X)
for i, clase in enumerate(modelo.classes):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=[p[clase] for p in probas], cmap='coolwarm', s=50, alpha=0.7)
    plt.colorbar(label=f"Probabilidad de {clase}")
    plt.title(f"Visualización de la probabilidad de {clase}")
    plt.xlabel("Length")
    plt.ylabel("Weight")
    plt.show()

# Ejemplo de predicción con un caso nuevo
predecir_especie(7, 6.3, 0.9)

# Ejemplo de predicción con un caso conocido
predecir_especie(10.66, 3.45, 0.32)
