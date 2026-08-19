import numpy as np
import sys
import os
import pickle

# Add the project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# pyrefly: ignore [missing-import]
from training.train.train_model.logistic_regression import LogisticRegression
# pyrefly: ignore [missing-import]
from training.train.train_model.svm import SVM
# pyrefly: ignore [missing-import]
from training.train.train_model.naive_bayes import MultinomialNaiveBayes

def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

def load_data(filepath):
    data = np.load(filepath)
    return data['X'], data['y']

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
    os.makedirs(models_dir, exist_ok=True)
    
    X_train, y_train = load_data(os.path.join(data_dir, 'train_data.npz'))
    X_val, y_val = load_data(os.path.join(data_dir, 'val_data.npz'))
    X_test, y_test = load_data(os.path.join(data_dir, 'test_data.npz'))
    
    print("Data loaded successfully.")
    print(f"Train size: {X_train.shape}, Val size: {X_val.shape}, Test size: {X_test.shape}")
    
    results = {}
    
    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(learning_rate=0.1, epochs=500)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results['Logistic Regression'] = calculate_metrics(y_test, lr_pred)
    
    with open(os.path.join(models_dir, 'logistic_regression.pkl'), 'wb') as f:
        pickle.dump(lr, f)
    
    # 2. SVM
    print("\nTraining SVM...")
    svm = SVM(learning_rate=0.01, lambda_param=0.01, epochs=500)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    results['SVM'] = calculate_metrics(y_test, svm_pred)
    
    with open(os.path.join(models_dir, 'svm.pkl'), 'wb') as f:
        pickle.dump(svm, f)
    
    # 3. Naive Bayes
    print("\nTraining Naive Bayes...")
    nb = MultinomialNaiveBayes()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    results['Naive Bayes'] = calculate_metrics(y_test, nb_pred)
    
    with open(os.path.join(models_dir, 'naive_bayes.pkl'), 'wb') as f:
        pickle.dump(nb, f)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS (TEST SET)")
    print("="*50)
    for model_name, metrics in results.items():
        acc, prec, rec, f1 = metrics
        print(f"[{model_name}]")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    main()
