"""
SQL Injection Detection using NLP - Query Preprocessor
Project by RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Contact: help@rskworld.in | support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in

This module handles preprocessing of SQL queries for feature extraction.
"""

import re
import string
from typing import List, Dict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class SQLQueryPreprocessor:
    """
    Preprocesses SQL queries for feature extraction and analysis.
    Handles normalization, tokenization, and cleaning of SQL queries.
    """
    
    def __init__(self):
        """Initialize the preprocessor with stopwords"""
        self.stop_words = set(stopwords.words('english'))
        # SQL-specific stopwords to keep
        self.sql_keywords = {
            'select', 'from', 'where', 'insert', 'update', 'delete',
            'drop', 'create', 'alter', 'table', 'database', 'union',
            'or', 'and', 'like', 'order', 'by', 'group', 'having',
            'join', 'inner', 'outer', 'left', 'right', 'on'
        }
        # Remove SQL keywords from stopwords
        self.stop_words = self.stop_words - self.sql_keywords
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize SQL query by converting to lowercase and removing extra whitespace.
        
        Args:
            query: Raw SQL query string
            
        Returns:
            Normalized query string
        """
        if not query or not isinstance(query, str):
            return ""
        
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove leading/trailing whitespace
        query = query.strip()
        
        return query
    
    def remove_special_chars(self, query: str, keep_sql_chars: bool = True) -> str:
        """
        Remove special characters from query while preserving SQL syntax.
        
        Args:
            query: SQL query string
            keep_sql_chars: If True, keep SQL-specific characters like quotes, parentheses
            
        Returns:
            Query with special characters removed
        """
        if keep_sql_chars:
            # Keep SQL-specific characters
            pattern = r'[^\w\s\'\"\(\)\-\+\*\/\=\<\>\;\,\#\@]'
        else:
            # Remove all special characters except spaces
            pattern = r'[^\w\s]'
        
        return re.sub(pattern, ' ', query)
    
    def tokenize(self, query: str) -> List[str]:
        """
        Tokenize SQL query into words.
        
        Args:
            query: SQL query string
            
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(query)
            return tokens
        except Exception as e:
            # Fallback to simple split if tokenization fails
            return query.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokenized query.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens without stopwords
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def preprocess(self, query: str, remove_stopwords_flag: bool = False) -> Dict:
        """
        Complete preprocessing pipeline for SQL query.
        
        Args:
            query: Raw SQL query string
            remove_stopwords_flag: Whether to remove stopwords
            
        Returns:
            Dictionary containing preprocessed query and tokens
        """
        # Normalize query
        normalized = self.normalize_query(query)
        
        # Tokenize
        tokens = self.tokenize(normalized)
        
        # Optionally remove stopwords
        if remove_stopwords_flag:
            tokens = self.remove_stopwords(tokens)
        
        return {
            'original': query,
            'normalized': normalized,
            'tokens': tokens,
            'token_count': len(tokens),
            'char_count': len(query),
            'word_count': len(tokens)
        }
    
    def extract_sql_keywords(self, query: str) -> List[str]:
        """
        Extract SQL keywords from query.
        
        Args:
            query: SQL query string
            
        Returns:
            List of SQL keywords found in query
        """
        normalized = self.normalize_query(query)
        tokens = self.tokenize(normalized)
        
        sql_keywords_found = [token for token in tokens if token in self.sql_keywords]
        return sql_keywords_found
    
    def detect_sql_operators(self, query: str) -> List[str]:
        """
        Detect SQL operators in query.
        
        Args:
            query: SQL query string
            
        Returns:
            List of SQL operators found
        """
        operators = ['=', '!=', '<>', '<', '>', '<=', '>=', 'like', 'in', 'not', 'and', 'or']
        found_operators = []
        
        normalized = self.normalize_query(query)
        
        for op in operators:
            if op in normalized:
                found_operators.append(op)
        
        return found_operators

