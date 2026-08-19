import numpy as np

class MultinomialNaiveBayes:
    def __init__(self):
        """
        Initialize the Multinomial Naive Bayes model.
        Suitable for discrete counts or TF-IDF continuous feature values.
        """
        self.classes = None
        self.class_log_prior = None
        self.feature_log_prob = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Naive Bayes classifier according to X, y.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.class_log_prior = np.zeros(n_classes)
        self.feature_log_prob = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            
            # log P(y)
            self.class_log_prior[idx] = np.log(X_c.shape[0] / n_samples)
            
            # Smoothing (alpha = 1.0)
            smoothed_cc = np.sum(X_c, axis=0) + 1.0
            smoothed_cc_sum = np.sum(smoothed_cc)
            
            # log P(x_i | y)
            self.feature_log_prob[idx, :] = np.log(smoothed_cc / smoothed_cc_sum)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict log probability of classes for each sample in X.
        
        Args:
            X (np.ndarray): Test data of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Log probability of each class, shape (n_samples, n_classes).
        """
        return np.dot(X, self.feature_log_prob.T) + self.class_log_prior

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on an array of test vectors X.
        
        Args:
            X (np.ndarray): Test data of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Predicted class labels.
        """
        log_prob = self.predict_log_proba(X)
        return self.classes[np.argmax(log_prob, axis=1)]
