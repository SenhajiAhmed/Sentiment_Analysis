import pandas as pd
import re
import contractions
import demoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

demoji.download_codes()

df = pd.read_csv('IMDB.csv')

def clean_data(df):
    """Perform essential data cleaning"""
    # Drop empty columns
    df.dropna(axis=1, how='all', inplace=True)
    
    # Drop rows with missing values in key columns
    df.dropna(subset=['review', 'sentiment'], inplace=True)
    
    # Remove duplicate reviews
    df = df.drop_duplicates(subset=['review'])
    
    return df

df = clean_data(df)

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

print("\nPreprocessing text...")
tqdm.pandas(desc="Cleaning reviews")
df['cleaned_review'] = df['review'].progress_apply(preprocess_text)

df.to_csv('cleaned_reviews.csv', index=False)