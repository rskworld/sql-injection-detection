# SQL Injection Detection using NLP

**Project by RSK World**  
**Founder:** Molla Samser  
**Designer & Tester:** Rima Khatun  
**Contact:** help@rskworld.in | support@rskworld.in  
**Phone:** +91 93305 39277  
**Location:** Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147  
**Website:** https://rskworld.in

## Quick Start

This project uses natural language processing and machine learning to detect SQL injection attacks by analyzing query patterns, syntax, and malicious payloads.

### Installation

```bash
pip install -r requirements.txt
python setup_nltk.py
```

### Usage

```bash
# Train the model
python train_model.py

# Detect SQL injection
python detect_sql_injection.py "SELECT * FROM users WHERE id = 1 OR 1=1"

# Interactive demo
python demo.py
```

## Features

- SQL query preprocessing
- Feature extraction (70+ features)
- NLP-based classification
- Attack pattern recognition
- Real-time detection system

## Technologies

Python, Scikit-learn, NLTK, TensorFlow, Pandas, Jupyter Notebook

## Documentation

📖 **For complete documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)**

The complete documentation includes:
- Detailed installation instructions
- Complete project structure
- Training data summary (134 samples)
- Feature extraction details (70+ features)
- Model training guide
- Usage examples
- And much more!

## License

This project is for educational purposes only.

## Disclaimer

Content used for educational purposes only. View Disclaimer at https://rskworld.in

