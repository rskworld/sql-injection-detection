# SQL Injection Detection using NLP - Complete Documentation

**Project by RSK World**  
**Founder:** Molla Samser  
**Designer & Tester:** Rima Khatun  
**Contact:** help@rskworld.in | support@rskworld.in  
**Phone:** +91 93305 39277  
**Location:** Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147  
**Website:** https://rskworld.in

---

## Table of Contents

1. [Project Description](#project-description)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Training Data Summary](#training-data-summary)
8. [Feature Extraction Summary](#feature-extraction-summary)
9. [Model Training](#model-training)
10. [License & Disclaimer](#license--disclaimer)

---

## Project Description

This project uses natural language processing and machine learning to detect SQL injection attacks by analyzing query patterns, syntax, and malicious payloads. It helps protect web applications from SQL injection vulnerabilities.

The system employs advanced feature extraction techniques to identify various SQL injection attack patterns, including tautology attacks, UNION-based injections, time-based blind SQL injection, error-based injection, and many more sophisticated attack vectors.

---

## Features

- ✅ **SQL query preprocessing** - Normalizes and cleans SQL queries for analysis
- ✅ **Feature extraction from queries** - Extracts 70+ features including lexical, syntactic, and pattern-based features
- ✅ **NLP-based classification** - Uses machine learning models to classify queries as safe or malicious
- ✅ **Attack pattern recognition** - Detects various SQL injection techniques and obfuscation methods
- ✅ **Real-time detection system** - Provides instant SQL injection detection for live queries

---

## Technologies Used

- **Python** - Core programming language
- **Scikit-learn** - Machine learning library for model training and evaluation
- **NLTK** - Natural Language Toolkit for text processing
- **TensorFlow** - Deep learning framework (optional for advanced models)
- **Pandas** - Data manipulation and analysis
- **Jupyter Notebook** - Interactive analysis and experimentation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone or Download Repository

Download or clone this repository to your local machine.

### Step 2: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

### Step 3: Setup NLTK Data

Download required NLTK datasets (run once):

```bash
python setup_nltk.py
```

Or manually in Python:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

---

## Usage

### Training the Model

Train the SQL injection detection model using the training dataset:

```bash
python train_model.py
```

This will:
- Load training data from `data/training_data.csv`
- Extract features from all queries
- Train multiple ML models (Random Forest, Gradient Boosting, Logistic Regression, SVM)
- Evaluate and select the best model
- Save the trained model to `models/sql_injection_model.pkl`

### Using the Detection System (Command Line)

Detect SQL injection in a single query:

```bash
python detect_sql_injection.py "SELECT * FROM users WHERE id = 1 OR 1=1"
```

### Interactive Demo

Run the interactive demo for testing multiple queries:

```bash
python demo.py
```

The demo allows you to:
- Test queries interactively
- See detailed analysis results
- View confidence scores
- Examine suspicious features

### Jupyter Notebook Analysis

For detailed analysis and experimentation:

1. Open Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to `notebooks/sql_injection_analysis.ipynb`

3. Run the notebook cells to:
   - Explore the dataset
   - Visualize feature distributions
   - Train and evaluate models
   - Test detection on sample queries

### Programmatic Usage

```python
from src.detector import SQLInjectionDetector

# Initialize detector
detector = SQLInjectionDetector()

# Detect SQL injection
result = detector.detect("SELECT * FROM users WHERE id = 1 OR 1=1")

if result['is_injection']:
    print("SQL Injection detected!")
    print(f"Confidence: {result['confidence']:.2%}")
else:
    print("Safe query")
```

---

## Project Structure

```
sql-injection-detection/
├── README.md                    # Quick start guide
├── DOCUMENTATION.md             # Complete documentation (this file)
├── PROJECT_STRUCTURE.md         # Detailed structure (merged here)
├── FEATURES_AND_DATA.md         # Features and data details (merged here)
├── requirements.txt             # Python dependencies
├── setup_nltk.py                # NLTK data download script
├── train_model.py               # Main training script
├── detect_sql_injection.py     # Command-line detection script
├── demo.py                     # Interactive demo script
├── .gitignore                   # Git ignore rules
├── sql-injection-detection.png # Project image placeholder
│
├── models/                      # Trained models directory
│   ├── sql_injection_model.pkl # Trained ML model (generated)
│   └── scaler.pkl               # Feature scaler (generated)
│
├── data/                        # Dataset directory
│   ├── training_data.csv        # Training dataset (134 samples)
│   └── test_data.csv           # Test dataset
│
├── src/                         # Source code package
│   ├── __init__.py              # Package initialization
│   ├── preprocessor.py          # SQL query preprocessing module
│   ├── feature_extractor.py     # Feature extraction for ML (70+ features)
│   ├── model_trainer.py         # Model training and evaluation
│   └── detector.py              # Real-time detection system
│
└── notebooks/                   # Jupyter notebooks
    └── sql_injection_analysis.ipynb  # Analysis and experimentation notebook
```

### Root Directory Files

- **README.md** - Quick start guide and basic information
- **DOCUMENTATION.md** - Complete documentation (this file)
- **requirements.txt** - Python package dependencies
- **setup_nltk.py** - Script to download NLTK datasets
- **train_model.py** - Main script for training the ML model
- **detect_sql_injection.py** - Command-line tool for detection
- **demo.py** - Interactive demo application
- **.gitignore** - Git version control ignore rules
- **sql-injection-detection.png** - Project image placeholder

### Source Code (`src/`)

- **`__init__.py`** - Package initialization file
- **`preprocessor.py`** - Handles SQL query preprocessing, normalization, and tokenization
- **`feature_extractor.py`** - Extracts 70+ features from SQL queries (lexical, syntactic, pattern-based)
- **`model_trainer.py`** - Trains and evaluates multiple ML models, selects best model
- **`detector.py`** - Real-time SQL injection detection system with detailed analysis

### Data (`data/`)

- **`training_data.csv`** - Training dataset with 134 SQL queries (55 safe, 79 injection)
- **`test_data.csv`** - Test dataset for model evaluation

### Models (`models/`)

- Directory created automatically after training
- **`sql_injection_model.pkl`** - Trained machine learning model (saved after training)
- **`scaler.pkl`** - Feature scaler for normalization (saved after training)

### Notebooks (`notebooks/`)

- **`sql_injection_analysis.ipynb`** - Jupyter notebook for data analysis, model training visualization, and experimentation

---

## Training Data Summary

### Dataset Statistics

- **Total Samples:** 134 SQL queries
- **Safe Queries (Label 0):** 55 samples (41%)
- **SQL Injection Queries (Label 1):** 79 samples (59%)

### Safe Query Types

The training dataset includes various legitimate SQL operations:

- **Basic SELECT queries** - Simple data retrieval
- **INSERT operations** - Data insertion with proper syntax
- **UPDATE operations** - Data modification queries
- **DELETE operations** - Data deletion queries
- **Complex queries with JOINs** - INNER, LEFT, RIGHT joins
- **Subqueries and nested queries** - Correlated and uncorrelated subqueries
- **SQL functions** - CONCAT, UPPER, LOWER, LENGTH, SUBSTRING, CAST, etc.
- **Aggregate functions** - COUNT, SUM, AVG, MAX, MIN
- **Filtering** - WHERE, HAVING, GROUP BY clauses
- **Sorting** - ORDER BY operations
- **Set operations** - IN, BETWEEN, LIKE, IS NULL, EXISTS

### SQL Injection Query Types

The dataset covers comprehensive SQL injection attack vectors:

#### Basic Attack Patterns
- **Tautology Attacks:** `OR 1=1`, `OR '1'='1'`, `AND 1=1`
- **Comment-based Attacks:** `--`, `#`, `/* */` comment injections
- **UNION Attacks:** `UNION SELECT` injections to extract data
- **Stacked Queries:** Multiple statements with semicolons (`; DROP TABLE;--`)

#### Advanced Injection Techniques
- **Time-based Blind SQL Injection:** `WAITFOR DELAY`, `SLEEP()`, `BENCHMARK()`
- **Command Execution:** `xp_cmdshell`, `sp_executesql`, `EXEC`
- **Boolean-based Blind:** Using `ASCII()`, `SUBSTRING()`, `LENGTH()` functions
- **Error-based Injection:** `EXTRACTVALUE()`, `UPDATEXML()`, `EXP()` functions
- **Encoding Attacks:** URL encoding (`%27`, `%3D`), hex encoding (`0x31`), char encoding (`CHAR(49)`)
- **Second-order Injection:** Nested subquery injections
- **Piggybacked Queries:** Multiple malicious statements chained together
- **Order/Group By Injection:** Injection in `ORDER BY`/`GROUP BY` clauses
- **Advanced Patterns:** Various obfuscation and evasion techniques

---

## Feature Extraction Summary

### Total Features: 70+

The feature extraction system analyzes SQL queries across three main categories:

### 1. Lexical Features (18 features)

**Basic Statistics:**
- `length` - Total character count
- `word_count` - Number of words
- `char_count` - Character count (same as length)

**Character Type Counts:**
- `digit_count` - Number of digits
- `letter_count` - Number of letters
- `special_char_count` - Number of special characters
- `space_count` - Number of spaces

**Case Features:**
- `uppercase_count` - Number of uppercase letters
- `lowercase_count` - Number of lowercase letters

**Ratio Features:**
- `digit_ratio` - Ratio of digits to total length
- `special_char_ratio` - Ratio of special characters
- `uppercase_ratio` - Ratio of uppercase letters
- `letter_ratio` - Ratio of letters
- `space_ratio` - Ratio of spaces

**Advanced Features:**
- `entropy` - Shannon entropy (randomness measure)
- `unique_chars` - Number of unique characters
- `most_common_char_freq` - Frequency of most common character
- `bigram_count` - Number of character bigrams

### 2. Syntactic Features (25 features)

**SQL Keyword Detection:**
- `sql_keyword_count` - Total SQL keywords found
- `has_select` - Contains SELECT keyword
- `has_union` - Contains UNION keyword
- `has_or` - Contains OR operator
- `has_and` - Contains AND operator
- `has_where` - Contains WHERE clause
- `has_drop` - Contains DROP keyword
- `has_delete` - Contains DELETE keyword
- `has_insert` - Contains INSERT keyword
- `has_update` - Contains UPDATE keyword
- `has_exec` - Contains EXEC/EXECUTE

**Operator Counts:**
- `equals_count` - Number of `=` operators
- `quote_count` - Number of quotes (`'` or `"`)
- `semicolon_count` - Number of semicolons
- `comment_count` - Number of comment markers
- `parentheses_count` - Number of parentheses

**Pattern Matching:**
- `has_comment` - Contains SQL comments
- `has_quotes` - Contains quotes
- `has_semicolon` - Contains semicolon
- `has_equals` - Contains equals operator

**SQL Functions:**
- `sql_function_count` - Total SQL functions found
- `has_count` - Contains COUNT function
- `has_concat` - Contains CONCAT function
- `has_cast` - Contains CAST/CONVERT

**Dangerous Functions:**
- `dangerous_function_count` - Count of dangerous functions
- `has_xp_cmdshell` - Contains xp_cmdshell or cmdshell

**Query Complexity:**
- `query_complexity` - Calculated complexity score
- `nested_level` - Nesting depth indicator

**Additional Operators:**
- `greater_than_count` - Number of `>` operators
- `less_than_count` - Number of `<` operators
- `not_equal_count` - Number of `!=` or `<>` operators
- `like_count` - Number of LIKE operators
- `in_count` - Number of IN operators

### 3. Pattern Features (27+ features)

**Basic Injection Patterns:**
- `injection_pattern_count` - Total injection patterns matched
- `has_tautology` - Contains tautology patterns (1=1, '1'='1')
- `has_union_attack` - UNION-based injection detected
- `has_comment_attack` - Comment-based injection
- `has_boolean_attack` - Boolean-based injection
- `has_time_delay` - Time-based delay functions
- `has_stacked_queries` - Multiple statements detected

**Encoding Patterns:**
- `has_url_encoding` - URL-encoded characters (%27, %3D)
- `has_hex_encoding` - Hexadecimal encoding (0x31)
- `has_suspicious_chars` - Suspicious character sequences

**Advanced Injection Types:**
- `has_blind_injection` - Blind SQL injection patterns
- `has_error_based` - Error-based injection techniques
- `has_second_order` - Second-order injection
- `has_piggybacked` - Piggybacked query attacks

**Encoding Variations:**
- `has_char_encoding` - CHAR() function encoding
- `has_unicode` - Unicode encoding patterns
- `has_double_encoding` - Double-encoded patterns

**Logic Patterns:**
- `has_always_true` - Always-true conditions
- `has_always_false` - Always-false conditions
- `has_null_injection` - NULL-based injection

**SQL-Specific Patterns:**
- `has_union_all` - UNION ALL statements
- `has_order_by_injection` - ORDER BY injection
- `has_group_by_injection` - GROUP BY injection

**Quote Manipulation:**
- `has_quote_escape` - Quote escaping patterns
- `has_unclosed_quote` - Unclosed quote detection

**Command Injection:**
- `has_command_injection` - Command execution patterns

### Feature Categories

#### Statistical Features
- Character frequency analysis
- Entropy calculation (randomness measure)
- N-gram analysis
- Ratio calculations for normalization

#### SQL-Specific Features
- SQL keyword detection and counting
- Function identification (safe and dangerous)
- Operator analysis
- Query structure and complexity analysis

#### Security Pattern Features
- Injection technique detection
- Encoding pattern recognition
- Attack vector identification
- Malicious pattern matching

---

## Model Training

### Training Process

The model training pipeline includes:

1. **Data Loading** - Loads training data from CSV
2. **Feature Extraction** - Extracts 70+ features from each query
3. **Data Preprocessing** - Splits data into train/test sets, scales features
4. **Model Training** - Trains multiple ML models:
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - Logistic Regression
   - Support Vector Machine (SVM)
5. **Model Evaluation** - Evaluates all models using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix
6. **Model Selection** - Selects best model based on F1-Score
7. **Model Saving** - Saves best model and scaler for deployment

### Benefits of Expanded Dataset and Features

The expanded dataset (134 samples) and enhanced features (70+) enable:

- **Better Generalization** - Model learns from diverse attack patterns
- **Improved Detection** - Better detection of obfuscated attacks
- **Higher Accuracy** - Improved performance on edge cases
- **Robust Performance** - More reliable real-world performance
- **Comprehensive Coverage** - Handles various SQL injection techniques

### Model Performance Metrics

After training, the system evaluates models on:
- **Accuracy** - Overall correctness
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1-Score** - Harmonic mean of precision and recall

The best model is selected based on F1-Score to balance precision and recall.

---

## License & Disclaimer

### License

This project is for **educational purposes only**.

### Disclaimer

Content used for educational purposes only. View Disclaimer at https://rskworld.in

### Ethical Use

This project is intended for:
- Educational purposes
- Security research
- Learning about SQL injection detection
- Improving application security

**Do not use this project for malicious purposes.**

---

## Support & Contact

**RSK World**  
**Founder:** Molla Samser  
**Designer & Tester:** Rima Khatun  

**Contact Information:**
- Email: help@rskworld.in | support@rskworld.in
- Phone: +91 93305 39277
- Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
- Website: https://rskworld.in

---

## Additional Resources

- **Jupyter Notebook:** `notebooks/sql_injection_analysis.ipynb` - Interactive analysis
- **Source Code:** `src/` directory - All implementation details
- **Training Data:** `data/training_data.csv` - Dataset for training
- **Requirements:** `requirements.txt` - All dependencies

---

**Last Updated:** 2025  
**Project Version:** 1.0.0  
**Status:** Active Development

---

*This documentation combines information from README.md, PROJECT_STRUCTURE.md, and FEATURES_AND_DATA.md into a single comprehensive guide.*

