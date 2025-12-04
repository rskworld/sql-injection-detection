"""
SQL Injection Detection using NLP - Model Trainer
Project by RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Contact: help@rskworld.in | support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in

This module handles training of machine learning models for SQL injection detection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple, Dict, Any
from src.feature_extractor import SQLFeatureExtractor


class SQLInjectionModelTrainer:
    """
    Trains and evaluates machine learning models for SQL injection detection.
    """
    
    def __init__(self):
        """Initialize the model trainer"""
        self.feature_extractor = SQLFeatureExtractor()
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.feature_names = self.feature_extractor.get_feature_names()
    
    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data from CSV file.
        
        Args:
            data_path: Path to CSV file with 'query' and 'label' columns
            
        Returns:
            Tuple of (features, labels)
        """
        df = pd.read_csv(data_path)
        
        # Extract features
        features = []
        labels = []
        
        for _, row in df.iterrows():
            query = row['query']
            label = row['label']
            
            feature_vector = self.feature_extractor.extract_all_features(query)
            features.append(feature_vector)
            labels.append(label)
        
        X = np.array(features)
        y = np.array(labels)
        
        return X, y
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for training by splitting and scaling.
        
        Args:
            X: Feature matrix
            y: Label vector
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train multiple ML models and return their performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        # Initialize models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        # Train all models
        trained_models = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        return trained_models
    
    def evaluate_models(self, models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate all models and return metrics.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'model': model
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
        
        return results
    
    def select_best_model(self, results: Dict[str, Dict]) -> Any:
        """
        Select the best model based on F1 score.
        
        Args:
            results: Dictionary of evaluation results
            
        Returns:
            Best model
        """
        best_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
        self.best_model = results[best_name]['model']
        
        print(f"\nBest Model: {best_name}")
        print(f"F1-Score: {results[best_name]['f1_score']:.4f}")
        
        return self.best_model
    
    def save_model(self, model: Any, model_path: str, scaler_path: str = None):
        """
        Save trained model and scaler to disk.
        
        Args:
            model: Trained model to save
            model_path: Path to save model
            scaler_path: Path to save scaler (optional)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        
        # Save model
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save scaler if provided
        if scaler_path and self.scaler:
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
    
    def train_and_save(self, data_path: str, model_path: str = 'models/sql_injection_model.pkl', 
                       scaler_path: str = 'models/scaler.pkl'):
        """
        Complete training pipeline: load data, train models, evaluate, and save best model.
        
        Args:
            data_path: Path to training data CSV
            model_path: Path to save best model
            scaler_path: Path to save scaler
        """
        print("Loading data...")
        X, y = self.load_data(data_path)
        print(f"Loaded {len(X)} samples")
        
        print("\nPreparing data...")
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        print("\nTraining models...")
        models = self.train_models(X_train, y_train)
        
        print("\nEvaluating models...")
        results = self.evaluate_models(models, X_test, y_test)
        
        print("\nSelecting best model...")
        best_model = self.select_best_model(results)
        
        print("\nSaving model...")
        self.save_model(best_model, model_path, scaler_path)
        
        return best_model, results

