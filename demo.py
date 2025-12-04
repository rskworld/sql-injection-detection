"""
SQL Injection Detection using NLP - Interactive Demo
Project by RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Contact: help@rskworld.in | support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in

Interactive demo script for SQL injection detection.
"""

import json
from src.detector import SQLInjectionDetector


def print_result(result):
    """Print detection result in formatted way"""
    print("\n" + "=" * 70)
    if result['is_injection']:
        print("⚠️  SQL INJECTION DETECTED")
    else:
        print("✅ Safe Query")
    print("=" * 70)
    
    print(f"\nConfidence: {result['confidence']:.2%}")
    print(f"\nQuery Analysis:")
    print(f"  Length: {result['details']['length']} characters")
    print(f"  Word Count: {result['details']['word_count']}")
    
    if result['details']['sql_keywords']:
        print(f"  SQL Keywords: {', '.join(result['details']['sql_keywords'])}")
    
    if result['details']['operators']:
        print(f"  Operators: {', '.join(result['details']['operators'])}")
    
    print(f"  Injection Patterns: {result['details']['injection_pattern_count']}")
    
    suspicious = result['details']['suspicious_features']
    if any(suspicious.values()):
        print(f"\n⚠️  Suspicious Features Detected:")
        for feature, value in suspicious.items():
            if value:
                print(f"    • {feature.replace('_', ' ').title()}")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """Main demo function"""
    print("=" * 70)
    print("SQL Injection Detection - Interactive Demo")
    print("Project by RSK World")
    print("=" * 70)
    print("\nThis demo allows you to test SQL queries for injection attacks.")
    print("Type 'exit' or 'quit' to end the demo.\n")
    
    # Initialize detector
    try:
        detector = SQLInjectionDetector()
    except Exception as e:
        print(f"Error initializing detector: {e}")
        print("Please ensure the model is trained first by running: python train_model.py")
        return
    
    # Sample queries for demonstration
    sample_queries = [
        ("SELECT * FROM users WHERE id = 1", "Safe query"),
        ("SELECT * FROM users WHERE id = 1 OR 1=1", "SQL injection - tautology"),
        ("SELECT * FROM users WHERE name = 'admin'--", "SQL injection - comment"),
        ("SELECT * FROM users UNION SELECT * FROM passwords", "SQL injection - union"),
        ("SELECT * FROM users WHERE id = 1; DROP TABLE users;--", "SQL injection - stacked queries"),
    ]
    
    print("Sample Queries (you can try these):")
    for i, (query, desc) in enumerate(sample_queries, 1):
        print(f"  {i}. {desc}")
        print(f"     {query}")
    print()
    
    # Interactive loop
    while True:
        try:
            query = input("Enter SQL query to analyze (or 'exit' to quit): ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nThank you for using SQL Injection Detection Demo!")
                break
            
            if not query:
                print("Please enter a valid query.\n")
                continue
            
            # Detect SQL injection
            result = detector.detect(query)
            
            if 'error' in result:
                print(f"Error: {result['error']}\n")
                continue
            
            # Print result
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n\nThank you for using SQL Injection Detection Demo!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

