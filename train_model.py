"""
SQL Injection Detection using NLP - Training Script
Project by RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Contact: help@rskworld.in | support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in

Main script to train the SQL injection detection model.
"""

import os
import sys
from src.model_trainer import SQLInjectionModelTrainer


def main():
    """Main training function"""
    print("=" * 60)
    print("SQL Injection Detection Model Training")
    print("Project by RSK World")
    print("=" * 60)
    
    # Check if data file exists
    data_path = 'data/training_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}")
        print("Please ensure the training data file exists.")
        sys.exit(1)
    
    # Initialize trainer
    trainer = SQLInjectionModelTrainer()
    
    # Train and save model
    model_path = 'models/sql_injection_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("\nStarting training process...")
    best_model, results = trainer.train_and_save(
        data_path=data_path,
        model_path=model_path,
        scaler_path=scaler_path
    )
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

