# One-hot-classification

import numpy as np
from sklearn.model_selection import train_test_split

X = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 0, 0],
              [0, 1, 0]])
y = np.array([[0], [1], [0], [0], [1]])  # Etiquetas binarias

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
