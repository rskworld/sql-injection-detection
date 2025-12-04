"""
SQL Injection Detection using NLP - Real-time Detector
Project by RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Contact: help@rskworld.in | support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in

This module provides real-time SQL injection detection functionality.
"""

import numpy as np
import joblib
import os
from typing import Dict, Tuple, Optional
from src.feature_extractor import SQLFeatureExtractor


class SQLInjectionDetector:
    """
    Real-time SQL injection detection system using trained ML model.
    """
    
    def __init__(self, model_path: str = 'models/sql_injection_model.pkl', 
                 scaler_path: str = 'models/scaler.pkl'):
        """
        Initialize detector with trained model.
        
        Args:
            model_path: Path to trained model file
            scaler_path: Path to scaler file
        """
        self.feature_extractor = SQLFeatureExtractor()
        self.model = None
        self.scaler = None
        
        # Load model and scaler if they exist
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}. Please train the model first.")
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
        else:
            print(f"Warning: Scaler not found at {scaler_path}.")
    
    def extract_features(self, query: str) -> np.ndarray:
        """
        Extract features from SQL query.
        
        Args:
            query: SQL query string
            
        Returns:
            Feature vector
        """
        return self.feature_extractor.extract_all_features(query)
    
    def predict(self, query: str) -> Tuple[int, float, Dict]:
        """
        Predict if query is SQL injection attack.
        
        Args:
            query: SQL query string to analyze
            
        Returns:
            Tuple of (prediction, confidence, details)
            - prediction: 0 (safe) or 1 (injection)
            - confidence: Probability score
            - details: Dictionary with feature analysis
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first.")
        
        # Extract features
        features = self.extract_features(query)
        features = features.reshape(1, -1)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features)[0]
        
        # Get prediction probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = max(probabilities)
        else:
            confidence = 1.0
        
        # Get detailed analysis
        details = self.analyze_query(query)
        details['prediction'] = 'SQL Injection' if prediction == 1 else 'Safe'
        details['confidence'] = confidence
        
        return prediction, confidence, details
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query and return detailed information.
        
        Args:
            query: SQL query string
            
        Returns:
            Dictionary with analysis details
        """
        # Extract different feature types
        lexical = self.feature_extractor.extract_lexical_features(query)
        syntactic = self.feature_extractor.extract_syntactic_features(query)
        pattern = self.feature_extractor.extract_pattern_features(query)
        
        # Get preprocessed info
        preprocessed = self.feature_extractor.preprocessor.preprocess(query)
        
        analysis = {
            'query': query,
            'length': lexical['length'],
            'word_count': lexical['word_count'],
            'sql_keywords': self.feature_extractor.preprocessor.extract_sql_keywords(query),
            'operators': self.feature_extractor.preprocessor.detect_sql_operators(query),
            'has_injection_patterns': pattern['injection_pattern_count'] > 0,
            'injection_pattern_count': pattern['injection_pattern_count'],
            'suspicious_features': {
                'has_union_attack': pattern['has_union_attack'],
                'has_comment_attack': pattern['has_comment_attack'],
                'has_boolean_attack': pattern['has_boolean_attack'],
                'has_time_delay': pattern['has_time_delay'],
                'has_stacked_queries': pattern['has_stacked_queries'],
                'has_url_encoding': pattern['has_url_encoding'],
            }
        }
        
        return analysis
    
    def detect(self, query: str) -> Dict:
        """
        Detect SQL injection with full analysis.
        
        Args:
            query: SQL query string
            
        Returns:
            Dictionary with detection results and analysis
        """
        try:
            prediction, confidence, details = self.predict(query)
            
            result = {
                'is_injection': bool(prediction == 1),
                'confidence': float(confidence),
                'details': details
            }
            
            return result
        except Exception as e:
            return {
                'error': str(e),
                'is_injection': None,
                'confidence': 0.0
            }
    
    def batch_detect(self, queries: list) -> list:
        """
        Detect SQL injection for multiple queries.
        
        Args:
            queries: List of SQL query strings
            
        Returns:
            List of detection results
        """
        results = []
        for query in queries:
            result = self.detect(query)
            results.append(result)
        return results

