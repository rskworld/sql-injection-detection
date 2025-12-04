# Release v1.0.0 - SQL Injection Detection using NLP

**Release Date:** 2025  
**Project by RSK World**  
**Founder:** Molla Samser  
**Designer & Tester:** Rima Khatun  

## 🎉 Initial Release

This is the first release of the SQL Injection Detection using NLP project.

## ✨ Features

- **70+ Feature Extraction** - Comprehensive feature set including lexical, syntactic, and pattern-based features
- **134 Training Samples** - Diverse dataset with 55 safe queries and 79 SQL injection queries
- **Multiple ML Models** - Random Forest, Gradient Boosting, Logistic Regression, and SVM
- **Real-time Detection** - Instant SQL injection detection system
- **Comprehensive Documentation** - Complete documentation with examples and guides

## 📦 What's Included

### Core Modules
- `src/preprocessor.py` - SQL query preprocessing and normalization
- `src/feature_extractor.py` - 70+ feature extraction system
- `src/model_trainer.py` - Model training and evaluation
- `src/detector.py` - Real-time detection system

### Scripts
- `train_model.py` - Train the ML model
- `detect_sql_injection.py` - Command-line detection tool
- `demo.py` - Interactive demo application
- `setup_nltk.py` - NLTK data setup script

### Data
- `data/training_data.csv` - 134 training samples
- `data/test_data.csv` - Test dataset

### Documentation
- `DOCUMENTATION.md` - Complete project documentation
- `README.md` - Quick start guide
- `notebooks/sql_injection_analysis.ipynb` - Jupyter analysis notebook

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Setup NLTK data
python setup_nltk.py

# Train the model
python train_model.py

# Test detection
python detect_sql_injection.py "SELECT * FROM users WHERE id = 1 OR 1=1"
```

## 📊 Dataset Statistics

- **Total Samples:** 134 queries
- **Safe Queries:** 55 samples
- **Injection Queries:** 79 samples
- **Features:** 70+ extracted features

## 🔍 Detection Capabilities

The system can detect:
- Tautology attacks (OR 1=1, OR '1'='1')
- Comment-based injections (--, #, /* */)
- UNION attacks
- Stacked queries
- Time-based blind SQL injection
- Boolean-based blind SQL injection
- Error-based SQL injection
- Encoding attacks (URL, hex, char)
- Second-order injection
- And many more attack vectors

## 🛠️ Technologies

- Python 3.7+
- Scikit-learn
- NLTK
- TensorFlow
- Pandas
- Jupyter Notebook

## 📝 Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for complete documentation.

## 📄 License

This project is for educational purposes only.

## 👥 Contributors

- **Founder:** Molla Samser
- **Designer & Tester:** Rima Khatun

## 📧 Contact

- Email: help@rskworld.in | support@rskworld.in
- Phone: +91 93305 39277
- Website: https://rskworld.in

---

**Download:** [Source Code (zip)](https://github.com/rskworld/sql-injection-detection/archive/refs/tags/v1.0.0.zip) | [Source Code (tar.gz)](https://github.com/rskworld/sql-injection-detection/archive/refs/tags/v1.0.0.tar.gz)

