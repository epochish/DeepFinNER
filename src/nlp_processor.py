"""
DeepFinNER: NLP Processor Module
Advanced Natural Language Processing for financial documents.

This module implements sophisticated NLP techniques that serve as
precursors to modern AI/ML applications in financial text analysis.
"""

import re
import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
from string import punctuation

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPProcessor:
    """
    Advanced NLP processor for financial documents.
    
    This class implements sophisticated text processing techniques that
    demonstrate early AI/ML capabilities in financial text analysis.
    """
    
    def __init__(self, custom_stopwords: Optional[Set[str]] = None):
        """
        Initialize the NLP processor.
        
        Args:
            custom_stopwords: Additional stopwords specific to financial domain
        """
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Financial domain stopwords
        financial_stopwords = {
            'company', 'corporation', 'inc', 'ltd', 'llc', 'corp',
            'million', 'billion', 'thousand', 'dollars', 'usd',
            'fiscal', 'year', 'quarter', 'period', 'ended',
            'total', 'net', 'gross', 'revenue', 'income', 'expense'
        }
        
        if custom_stopwords:
            financial_stopwords.update(custom_stopwords)
        
        self.stop_words.update(financial_stopwords)
        
        # Financial entity patterns
        self.financial_patterns = {
            'currency': r'\$[\d,]+(?:\.\d{2})?',
            'percentage': r'\d+(?:\.\d+)?%',
            'date': r'\d{1,2}/\d{1,2}/\d{2,4}',
            'year': r'\b(?:19|20)\d{2}\b',
            'quarter': r'Q[1-4]',
            'financial_metric': r'\b(?:revenue|profit|loss|earnings|assets|liabilities|equity)\b'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning for financial documents.
        
        This preprocessing step is crucial for NLP applications and
        demonstrates advanced text processing techniques.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags and entities
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s\$%\.\-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str, method: str = 'word') -> List[str]:
        """
        Advanced tokenization with multiple methods.
        
        Args:
            text: Input text
            method: Tokenization method ('word', 'sentence', 'paragraph')
            
        Returns:
            List of tokens
        """
        if method == 'word':
            tokens = word_tokenize(text)
        elif method == 'sentence':
            tokens = sent_tokenize(text)
        elif method == 'paragraph':
            tokens = [p.strip() for p in text.split('\n\n') if p.strip()]
        else:
            raise ValueError(f"Unknown tokenization method: {method}")
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens for better analysis."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stem tokens for root form analysis."""
        return [self.stemmer.stem(token) for token in tokens]
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial entities using pattern matching.
        
        This demonstrates early Named Entity Recognition techniques
        that paved the way for modern AI/ML applications.
        """
        entities = {}
        
        for entity_type, pattern in self.financial_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))
        
        return entities
    
    def calculate_text_statistics(self, text: str) -> Dict[str, float]:
        """
        Calculate comprehensive text statistics.
        
        Returns:
            Dictionary containing various text metrics
        """
        # Basic statistics
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Calculate metrics
        stats = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0
        }
        
        return stats
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Perform sentiment analysis on financial text.
        
        This demonstrates early sentiment analysis techniques that
        are now fundamental to modern AI/ML applications.
        """
        blob = TextBlob(text)
        
        # Polarity (-1 to 1, negative to positive)
        polarity = blob.sentiment.polarity
        
        # Subjectivity (0 to 1, objective to subjective)
        subjectivity = blob.sentiment.subjectivity
        
        # Categorize sentiment
        if polarity > 0.1:
            sentiment_category = 'positive'
        elif polarity < -0.1:
            sentiment_category = 'negative'
        else:
            sentiment_category = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment_category': sentiment_category
        }
    
    def create_word_frequency_matrix(self, documents: List[str], 
                                   max_features: int = 1000) -> Tuple[np.ndarray, List[str]]:
        """
        Create word frequency matrix for document analysis.
        
        This demonstrates early document vectorization techniques
        that are fundamental to modern NLP and AI applications.
        """
        # Clean and preprocess documents
        cleaned_docs = [self.clean_text(doc) for doc in documents]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        return tfidf_matrix.toarray(), feature_names
    
    def calculate_document_similarity(self, doc1: str, doc2: str) -> float:
        """
        Calculate cosine similarity between two documents.
        
        This demonstrates early document similarity techniques
        that are now fundamental to modern AI/ML applications.
        """
        # Clean documents
        cleaned_doc1 = self.clean_text(doc1)
        cleaned_doc2 = self.clean_text(doc2)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([cleaned_doc1, cleaned_doc2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity
    
    def extract_key_phrases(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract key phrases using TF-IDF scoring.
        
        This demonstrates early keyword extraction techniques
        that are now fundamental to modern AI/ML applications.
        """
        # Clean and tokenize
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text, 'word')
        tokens = self.remove_stopwords(tokens)
        
        # Create bigrams
        bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        
        # Combine unigrams and bigrams
        all_phrases = tokens + bigrams
        
        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=1000
        )
        
        tfidf_matrix = vectorizer.fit_transform([cleaned_text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get scores
        scores = tfidf_matrix.toarray()[0]
        
        # Create phrase-score pairs
        phrase_scores = list(zip(feature_names, scores))
        
        # Sort by score and return top k
        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        
        return phrase_scores[:top_k]
    
    def process_financial_document(self, text: str) -> Dict:
        """
        Comprehensive processing of financial documents.
        
        This method demonstrates a complete NLP pipeline that
        showcases early AI/ML capabilities in financial text analysis.
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        words = self.tokenize_text(cleaned_text, 'word')
        sentences = self.tokenize_text(cleaned_text, 'sentence')
        
        # Remove stopwords
        filtered_words = self.remove_stopwords(words)
        
        # Lemmatize
        lemmatized_words = self.lemmatize_tokens(filtered_words)
        
        # Extract entities
        entities = self.extract_financial_entities(text)
        
        # Calculate statistics
        stats = self.calculate_text_statistics(text)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(text)
        
        # Extract key phrases
        key_phrases = self.extract_key_phrases(text)
        
        # Create comprehensive results
        results = {
            'text_statistics': stats,
            'sentiment_analysis': sentiment,
            'financial_entities': entities,
            'key_phrases': key_phrases,
            'word_frequency': Counter(lemmatized_words).most_common(20),
            'processing_metadata': {
                'original_length': len(text),
                'cleaned_length': len(cleaned_text),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'unique_words': len(set(words))
            }
        }
        
        return results


def batch_process_documents(documents: List[Dict], processor: NLPProcessor) -> List[Dict]:
    """
    Process multiple documents in batch.
    
    Args:
        documents: List of document dictionaries with 'text' key
        processor: NLPProcessor instance
        
    Returns:
        List of processed document results
    """
    results = []
    
    for doc in documents:
        try:
            processed = processor.process_financial_document(doc['text'])
            processed['document_id'] = doc.get('id', 'unknown')
            processed['document_date'] = doc.get('date', 'unknown')
            results.append(processed)
        except Exception as e:
            logger.error(f"Error processing document {doc.get('id', 'unknown')}: {e}")
            continue
    
    return results


if __name__ == "__main__":
    # Example usage
    processor = NLPProcessor()
    
    # Sample financial text
    sample_text = """
    Apple Inc. reported revenue of $394.3 billion for fiscal year 2022, 
    representing an 8% increase compared to the previous year. The company's 
    net income reached $99.8 billion, driven by strong iPhone sales and 
    services growth. However, supply chain challenges and macroeconomic 
    uncertainty pose risks to future performance.
    """
    
    # Process the text
    results = processor.process_financial_document(sample_text)
    
    print("Text Statistics:", results['text_statistics'])
    print("Sentiment Analysis:", results['sentiment_analysis'])
    print("Financial Entities:", results['financial_entities'])
    print("Key Phrases:", results['key_phrases'][:5]) 