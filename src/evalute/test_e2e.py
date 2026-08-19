"""
End-to-End Test Script
======================
Simulates a real-world scenario by taking a raw email string,
running it through the exact preprocessing pipeline (cleaning, feature extraction, TF-IDF),
and using the best trained model (Naive Bayes) to predict if it is Spam or Ham.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import re
import pickle
import numpy as np

# Set paths
BASE_DIR = r"d:\Machine Learning\máy học\PRE1 - Email Classfication"
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 1. Reuse the exact cleaning logic from step 1
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'it', 'was', 'be', 'are', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'not',
    'no', 'if', 'as', 'so', 'than', 'then', 'just', 'about', 'up',
    'out', 'all', 'more', 'also', 'very', 'what', 'when', 'where',
    'how', 'who', 'which', 'there', 'subject', 're', 'fw', 'any', 'been', 
    'each', 'new', 'one', 'two', 'get', 'like', 'know', 'please', 'would', 
    'need', 'am', 'pm'
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    text = re.sub(r'^subject:\s*', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return " ".join(words)

def extract_features(raw_text: str):
    excl_count = raw_text.count('!')
    word_count = len(raw_text.split())
    clean = clean_text(raw_text)
    return clean, excl_count, word_count

# 2. Reuse TF-IDF transformation logic from step 3
def transform_tfidf(docs, vocab, idf):
    n_docs = len(docs)
    vocab_size = len(vocab)
    matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)
    
    for i, doc in enumerate(docs):
        tf = {}
        for w in doc:
            if w in vocab:
                tf[w] = tf.get(w, 0) + 1
        for w, count in tf.items():
            idx = vocab[w]
            matrix[i, idx] = count * idf[idx]
            
        norm = np.linalg.norm(matrix[i])
        if norm > 0:
            matrix[i] = matrix[i] / norm
            
    return matrix

def predict_email(raw_email: str, model_name='naive_bayes.pkl'):
    print(f"\n--- Testing E2E Pipeline with: {model_name} ---")
    print(f"Raw Input: '{raw_email}'")
    
    # Extract features
    clean, excl_count, word_count = extract_features(raw_email)
    print(f"Cleaned Text: '{clean}'")
    
    # Load vocabulary
    with open(os.path.join(PROC_DIR, 'tfidf_vocab.pkl'), 'rb') as f:
        vocab_data = pickle.load(f)
    vocab = vocab_data['vocab']
    idf = vocab_data['idf']
    
    # Transform text
    X_text = transform_tfidf([clean.split()], vocab, idf)
    
    # Prepare numerical features (log1p normalized as in training)
    X_num = np.array([[np.log1p(excl_count), np.log1p(word_count)]], dtype=np.float32)
    
    # Combine
    X_final = np.hstack([X_text, X_num])
    
    # Load model
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    with open(os.path.join(MODELS_DIR, model_name), 'rb') as f:
        model = pickle.load(f)
        
    # Predict
    prediction = model.predict(X_final)[0]
    result = "SPAM" if prediction == 1 else "HAM (Not Spam)"
    print(f">>> Prediction: {result}")
    return result

if __name__ == "__main__":
    # Test case 1: Typical Spam
    spam_email = "URGENT: You have won a FREE $1000 Walmart Gift Card! Click here now to claim your prize! Offer ends soon!!"
    predict_email(spam_email, 'naive_bayes.pkl')
    
    # Test case 2: Typical Ham
    ham_email = "Subject: Q3 Financial Report Review. Hi team, please find attached the Q3 financial report. We will discuss the budget allocations in tomorrow's meeting. Thanks, Vince."
    predict_email(ham_email, 'naive_bayes.pkl')
    
    print("\nEnd-to-End Test Completed Successfully!")
