"""
SQL Injection Detection using NLP - Detection Script
Project by RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Contact: help@rskworld.in | support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in

Command-line script for detecting SQL injection in queries.
"""

import sys
import json
from src.detector import SQLInjectionDetector


def main():
    """Main detection function"""
    if len(sys.argv) < 2:
        print("Usage: python detect_sql_injection.py <sql_query>")
        print("\nExample:")
        print('  python detect_sql_injection.py "SELECT * FROM users WHERE id = 1"')
        print('  python detect_sql_injection.py "SELECT * FROM users WHERE id = 1 OR 1=1"')
        sys.exit(1)
    
    query = sys.argv[1]
    
    print("=" * 60)
    print("SQL Injection Detection")
    print("Project by RSK World")
    print("=" * 60)
    print(f"\nAnalyzing query: {query}\n")
    
    # Initialize detector
    detector = SQLInjectionDetector()
    
    # Detect SQL injection
    result = detector.detect(query)
    
    # Display results
    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    print("Detection Results:")
    print("-" * 60)
    print(f"Status: {'⚠️  SQL INJECTION DETECTED' if result['is_injection'] else '✅ Safe Query'}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nDetailed Analysis:")
    print(f"  Query Length: {result['details']['length']} characters")
    print(f"  Word Count: {result['details']['word_count']}")
    print(f"  SQL Keywords: {', '.join(result['details']['sql_keywords']) if result['details']['sql_keywords'] else 'None'}")
    print(f"  Operators: {', '.join(result['details']['operators']) if result['details']['operators'] else 'None'}")
    print(f"  Injection Patterns Found: {result['details']['injection_pattern_count']}")
    
    if result['details']['suspicious_features']:
        print(f"\nSuspicious Features:")
        for feature, value in result['details']['suspicious_features'].items():
            if value:
                print(f"  ⚠️  {feature.replace('_', ' ').title()}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

