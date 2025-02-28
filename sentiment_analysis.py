# -*- coding: utf-8 -*-
import pandas as pd
import re
import contractions
import demoji
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, ConfusionMatrixDisplay)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.svm import LinearSVC


# Download required NLP resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

demoji.download_codes()

# --------------------------
# Data Loading & Cleaning
# --------------------------
def clean_data(df):
    """Perform essential data cleaning"""
    # Drop empty columns
    df.dropna(axis=1, how='all', inplace=True)
    
    # Drop rows with missing values in key columns
    df.dropna(subset=['review', 'sentiment'], inplace=True)
    
    # Remove duplicate reviews
    df = df.drop_duplicates(subset=['review'])
    
    return df

# Load and clean data
df = pd.read_csv('IMDB.csv')
df = clean_data(df)

# ------------------
# Text Preprocessing
# ------------------
def preprocess_text(text, handle_negations=True):
    """Enhanced text cleaning with negation handling"""
    # Initial cleaning
    text = re.sub(r'http\S+|www\S+|https\S+|<.*?>', '', text)
    text = contractions.fix(text)
    text = demoji.replace_with_desc(text, sep=" ")
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Negation handling
    if handle_negations:
        negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing'}
        negated = False
        processed_tokens = []
        
        for token in tokens:
            if token in negation_words:
                negated = True
                processed_tokens.append(token)
            else:
                if negated:
                    processed_tokens.append(f"NOT_{token}")
                    negated = False  # Reset after next word
                else:
                    processed_tokens.append(token)
    
    # Stopword removal and lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    final_tokens = [
        lemmatizer.lemmatize(token) 
        for token in (processed_tokens if handle_negations else tokens)
        if token not in stop_words and len(token) > 2
    ]
    
    return ' '.join(final_tokens)

# Apply preprocessing with progress bar
print("\nPreprocessing text...")
tqdm.pandas(desc="Cleaning reviews")
df['cleaned_review'] = df['review'].progress_apply(preprocess_text)

# ------------------------
# Data Splitting & Vectorization
# ------------------------
# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    df['cleaned_review'],
    df['sentiment'],
    test_size=0.2,
    stratify=df['sentiment'],
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

# TF-IDF Vectorization with progress
print("\nVectorizing text...")
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english'
)

with tqdm(total=3, desc="TF-IDF Processing") as pbar:
    X_train_tfidf = tfidf.fit_transform(X_train)
    pbar.update(1)
    X_val_tfidf = tfidf.transform(X_val)
    pbar.update(1)
    X_test_tfidf = tfidf.transform(X_test)
    pbar.update(1)

# ------------------
# Model Training with Progress
# ------------------
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'SVM': LinearSVC(C=1.0, class_weight='balanced', max_iter=1000) #, class_weight='balanced', probability=True
} 

results = []
model_objects = {}

for model_name, model in models.items():
    with tqdm(total=100, desc=f"Training {model_name}", ncols=100) as pbar:
        # Training phase
        model.fit(X_train_tfidf, y_train)
        pbar.update(70)  # Simulating 70% progress for training
        
        # Validation phase
        y_val_pred = model.predict(X_val_tfidf)
        pbar.update(20)  # 20% for validation
        
        # Metrics calculation
        report = classification_report(y_val, y_val_pred, output_dict=True)
        pbar.update(10)  # 10% for metrics
        
        # Store results
        results.append({
            'Model': model_name,
            'Validation Accuracy': accuracy_score(y_val, y_val_pred),
            'Classification Report': report
        })
        model_objects[model_name] = model

    # Print metrics
    print(f"\n{model_name} Validation Results:")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(classification_report(y_val, y_val_pred))

# ------------------
# Results Comparison
# ------------------
# Create comparison DataFrame
comparison_df = pd.DataFrame(results)

# Extract precision/recall/f1 metrics
metrics = []
for result in results:
    report = result['Classification Report']
    metrics.append({
        'Model': result['Model'],
        'Accuracy': result['Validation Accuracy'],
        'Precision (Positive)': report['positive']['precision'],
        'Recall (Positive)': report['positive']['recall'],
        'F1 (Positive)': report['positive']['f1-score'],
        'Precision (Negative)': report['negative']['precision'],
        'Recall (Negative)': report['negative']['recall'],
        'F1 (Negative)': report['negative']['f1-score']
    })

metric_df = pd.DataFrame(metrics).set_index('Model')
print("\nModel Comparison Metrics:")
print(metric_df.T)

# Visual comparison
plt.figure(figsize=(10, 6))
metric_df[['Accuracy']].plot(kind='bar', rot=0)
plt.title('Model Accuracy Comparison')
plt.ylabel('Score')
plt.ylim(0.7, 1.0)
plt.show()

# ------------------
# Final Test Evaluation
# ------------------
# Select best model based on validation
best_model_name = max(results, key=lambda x: x['Validation Accuracy'])['Model']
best_model = models[best_model_name]

print(f"\n{'='*40}\nEvaluating Best Model ({best_model_name}) on Test Set\n{'='*40}")

# Test evaluation
y_test_pred = best_model.predict(X_test_tfidf)
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred))

# ------------------
# Model Persistence
# ------------------
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(best_model, 'best_model.pkl')

# ------------------
# Prediction Function
# ------------------
def predict_sentiment(text, model=best_model):
    """End-to-end sentiment prediction pipeline"""
    cleaned_text = preprocess_text(text)
    vectorized_text = tfidf.transform([cleaned_text])
    return model.predict(vectorized_text)[0]

# Sample predictions
test_reviews = [
    "This movie was an absolute masterpiece! The acting was superb.",
    "Terrible experience from start to finish. Waste of money.",
    "The product works okay, but nothing special for the price."
]

print("\nSample Predictions:")
for review in test_reviews:
    prediction = predict_sentiment(review)
    print(f"\nReview: {review}\nPredicted Sentiment: {prediction}")
    print("-"*60)