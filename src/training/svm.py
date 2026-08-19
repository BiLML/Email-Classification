import numpy as np

class SVM:
    def __init__(self, learning_rate: float = 0.01, lambda_param: float = 0.01, epochs: int = 1000):
        """
        Initialize the Support Vector Machine (Linear SVM) model.
        
        Args:
            learning_rate (float): The step size for gradient descent.
            lambda_param (float): Regularization parameter.
            epochs (int): The number of iterations over the training data.
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using gradient descent.
        SVM uses hinge loss and requires labels in {-1, 1}.
        If y is in {0, 1}, we convert it to {-1, 1}.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Convert {0, 1} labels to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.weights
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    db = y_[idx]
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the given test data.
        
        Args:
            X (np.ndarray): Test data of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Predicted labels in {0, 1}.
        """
        linear_model = np.dot(X, self.weights) - self.bias
        predictions = np.sign(linear_model)
        
        # Convert {-1, 1} back to {0, 1}
        return np.where(predictions == -1, 0, 1)
