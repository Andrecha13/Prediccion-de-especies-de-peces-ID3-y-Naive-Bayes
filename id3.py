import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("fish_data.csv")

# Separar características (X) y etiquetas (y)
X = df[['length', 'weight', 'w_l_ratio']].values
y = df['species'].values
features = ['length', 'weight', 'w_l_ratio']

# Implementación manual del algoritmo ID3
# Modificar la clase ID3Tree
class ID3Tree:
    def __init__(self):
        self.tree = None
        self.default_class = None

    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, X_column, y):
        total_entropy = self._entropy(y)
        values, counts = np.unique(X_column, return_counts=True)
        weighted_entropy = np.sum(
            (counts[i] / len(y)) * self._entropy(y[X_column == value])
            for i, value in enumerate(values)
        )
        return total_entropy - weighted_entropy

    def _best_split(self, X, y, features):
        best_gain = -1
        best_feature = None
        for i, feature in enumerate(features):
            gain = self._information_gain(X[:, i], y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature

    def _build_tree(self, X, y, features):
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]  # Hoja con clase única
        if len(features) == 0 or len(y) == 0:
            return self.default_class  # Clase mayoritaria como predeterminada

        best_feature = self._best_split(X, y, features)
        if best_feature is None:
            return self.default_class

        tree = {best_feature: {}}
        feature_index = features.index(best_feature)

        for value in np.unique(X[:, feature_index]):
            sub_X = X[X[:, feature_index] == value]
            sub_y = y[X[:, feature_index] == value]
            subtree = self._build_tree(
                sub_X, sub_y, [f for f in features if f != best_feature]
            )
            tree[best_feature][value] = subtree

        return tree

    def fit(self, X, y, features):
        self.default_class = np.bincount(y).argmax()  # Clase mayoritaria
        self.tree = self._build_tree(X, y, features)

    def _predict_instance(self, instance, tree):
        if not isinstance(tree, dict):  # Si es una hoja
            return tree
        feature = list(tree.keys())[0]
        feature_value = instance[features.index(feature)]
        if feature_value in tree[feature]:
            return self._predict_instance(instance, tree[feature][feature_value])
        else:
            return self.default_class  # Clase predeterminada

    def predict(self, X):
        return np.array([self._predict_instance(instance, self.tree) for instance in X])

# Convertir etiquetas a números
class_mapping = {label: i for i, label in enumerate(np.unique(y))}
y_encoded = np.array([class_mapping[label] for label in y])

# Entrenar el modelo
id3 = ID3Tree()
id3.fit(X, y_encoded, features)

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
    prediccion_codificada = id3.predict(nuevo_pez)
    prediccion = [list(class_mapping.keys())[list(class_mapping.values()).index(p)] for p in prediccion_codificada]
    print(f"\nPredicción: {prediccion[0]}")
    return prediccion[0]

# Graficar las características con colores por especie
sns.scatterplot(data=df, x='length', y='weight', hue='species', palette='viridis')
plt.xlabel("Length (cm)")
plt.ylabel("Weight (g)")
plt.title("Length vs Weight from Fish Species")
plt.show()

# Ejemplo de predicción con un caso nuevo
predecir_especie(7, 6.3, 0.9)

# Ejemplo de predicción con un caso conocido
predecir_especie(10.66, 3.45, 0.32)
