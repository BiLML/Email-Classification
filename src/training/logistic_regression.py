import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        """
        Initialize the Logistic Regression model.
        
        Args:
            learning_rate (float): The step size for gradient descent.
            epochs (int): The number of iterations over the training data.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid of z.
        
        Args:
            z (np.ndarray): Linear transformation (X * w + b).
            
        Returns:
            np.ndarray: Sigmoid of z.
        """
        # Clip z to prevent overflow in np.exp
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using gradient descent.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability of each sample belonging to the positive class.
        
        Args:
            X (np.ndarray): Test data of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples,).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict the class labels for the given test data.
        
        Args:
            X (np.ndarray): Test data of shape (n_samples, n_features).
            threshold (float): Decision threshold.
            
        Returns:
            np.ndarray: Predicted labels of shape (n_samples,).
        """
        y_predicted_cls = self.predict_proba(X) >= threshold
        return y_predicted_cls.astype(int)
