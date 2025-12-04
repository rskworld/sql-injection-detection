"""
SQL Injection Detection using NLP - Feature Extractor
Project by RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Contact: help@rskworld.in | support@rskworld.in
Phone: +91 93305 39277
Location: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in

This module extracts features from SQL queries for machine learning classification.
"""

import re
import numpy as np
import math
from typing import List, Dict, Any
from collections import Counter
from src.preprocessor import SQLQueryPreprocessor


class SQLFeatureExtractor:
    """
    Extracts various features from SQL queries for ML classification.
    Features include lexical, syntactic, and semantic characteristics.
    """
    
    def __init__(self):
        """Initialize feature extractor with preprocessor"""
        self.preprocessor = SQLQueryPreprocessor()
        
        # Common SQL injection patterns
        self.injection_patterns = [
            r"('|(\\')|(;)|(--)|(\*)|(\/\*)|(\*\/)|(\+)|(\-)|(\%)|(\=)|(\()|(\))|(\[)|(\])|(\{)|(\}))",
            r"(union|select|insert|update|delete|drop|create|alter|exec|execute)",
            r"(\bor\b|\band\b).*(\d+|\'|\")",
            r"(\'|\"|;|--|\*|\/\*|\*\/)",
            r"(1\s*=\s*1|1\s*=\s*\'1\'|\'1\'=\'1\')",
            r"(waitfor|delay|sleep|benchmark)",
            r"(xp_|sp_|cmdshell|shell)",
            r"(\%27|\%22|\%3D|\%3B|\%2D|\%2D)",
        ]
        
        # SQL keywords
        self.sql_keywords = {
            'select', 'from', 'where', 'insert', 'update', 'delete',
            'drop', 'create', 'alter', 'table', 'database', 'union',
            'or', 'and', 'like', 'order', 'by', 'group', 'having',
            'join', 'inner', 'outer', 'left', 'right', 'on', 'exec',
            'execute', 'script', 'waitfor', 'delay', 'sleep'
        }
        
        # SQL functions
        self.sql_functions = {
            'count', 'sum', 'avg', 'max', 'min', 'concat', 'substring',
            'char', 'ascii', 'cast', 'convert', 'isnull', 'coalesce',
            'upper', 'lower', 'trim', 'ltrim', 'rtrim', 'len', 'length'
        }
        
        # Dangerous SQL functions/commands
        self.dangerous_functions = {
            'xp_cmdshell', 'xp_regread', 'sp_executesql', 'openrowset',
            'bulk', 'bcp', 'shutdown', 'xp_dirtree', 'xp_fileexist',
            'xp_getfiledetails', 'sp_makewebtask', 'xp_cmdshell'
        }
    
    def extract_lexical_features(self, query: str) -> Dict[str, Any]:
        """
        Extract lexical features from SQL query.
        
        Args:
            query: SQL query string
            
        Returns:
            Dictionary of lexical features
        """
        features = {}
        
        # Basic statistics
        features['length'] = len(query)
        features['word_count'] = len(query.split())
        features['char_count'] = len(query)
        
        # Character type counts
        features['digit_count'] = sum(c.isdigit() for c in query)
        features['letter_count'] = sum(c.isalpha() for c in query)
        features['special_char_count'] = sum(not c.isalnum() and not c.isspace() for c in query)
        features['space_count'] = query.count(' ')
        
        # Case features
        features['uppercase_count'] = sum(c.isupper() for c in query)
        features['lowercase_count'] = sum(c.islower() for c in query)
        
        # Ratio features
        if features['length'] > 0:
            features['digit_ratio'] = features['digit_count'] / features['length']
            features['special_char_ratio'] = features['special_char_count'] / features['length']
            features['uppercase_ratio'] = features['uppercase_count'] / features['length']
            features['letter_ratio'] = features['letter_count'] / features['length']
            features['space_ratio'] = features['space_count'] / features['length']
        else:
            features['digit_ratio'] = 0
            features['special_char_ratio'] = 0
            features['uppercase_ratio'] = 0
            features['letter_ratio'] = 0
            features['space_ratio'] = 0
        
        # Entropy calculation (measure of randomness)
        features['entropy'] = self._calculate_entropy(query)
        
        # Character frequency features
        char_freq = Counter(query.lower())
        if len(char_freq) > 0:
            features['unique_chars'] = len(char_freq)
            features['most_common_char_freq'] = char_freq.most_common(1)[0][1] / len(query) if query else 0
        else:
            features['unique_chars'] = 0
            features['most_common_char_freq'] = 0
        
        # N-gram features (bigrams)
        features['bigram_count'] = len(query) - 1 if len(query) > 1 else 0
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of the text"""
        if not text:
            return 0.0
        char_freq = Counter(text.lower())
        length = len(text)
        entropy = -sum((count / length) * math.log2(count / length) 
                      for count in char_freq.values() if count > 0)
        return entropy
    
    def extract_syntactic_features(self, query: str) -> Dict[str, Any]:
        """
        Extract syntactic features from SQL query.
        
        Args:
            query: SQL query string
            
        Returns:
            Dictionary of syntactic features
        """
        features = {}
        normalized = self.preprocessor.normalize_query(query)
        
        # SQL keyword counts
        tokens = self.preprocessor.tokenize(normalized)
        sql_keyword_count = sum(1 for token in tokens if token in self.sql_keywords)
        features['sql_keyword_count'] = sql_keyword_count
        
        # Specific SQL keyword presence
        features['has_select'] = 'select' in normalized
        features['has_union'] = 'union' in normalized
        features['has_or'] = ' or ' in normalized or normalized.startswith('or ') or normalized.endswith(' or')
        features['has_and'] = ' and ' in normalized or normalized.startswith('and ') or normalized.endswith(' and')
        features['has_where'] = 'where' in normalized
        features['has_drop'] = 'drop' in normalized
        features['has_delete'] = 'delete' in normalized
        features['has_insert'] = 'insert' in normalized
        features['has_update'] = 'update' in normalized
        features['has_exec'] = 'exec' in normalized or 'execute' in normalized
        
        # Operator counts
        features['equals_count'] = normalized.count('=')
        features['quote_count'] = normalized.count("'") + normalized.count('"')
        features['semicolon_count'] = normalized.count(';')
        features['comment_count'] = normalized.count('--') + normalized.count('/*') + normalized.count('*/')
        features['parentheses_count'] = normalized.count('(') + normalized.count(')')
        
        # Pattern matching
        features['has_comment'] = '--' in normalized or '/*' in normalized
        features['has_quotes'] = "'" in normalized or '"' in normalized
        features['has_semicolon'] = ';' in normalized
        features['has_equals'] = '=' in normalized
        
        # SQL function detection
        tokens = self.preprocessor.tokenize(normalized)
        sql_function_count = sum(1 for token in tokens if token in self.sql_functions)
        features['sql_function_count'] = sql_function_count
        features['has_count'] = 'count' in normalized
        features['has_concat'] = 'concat' in normalized
        features['has_cast'] = 'cast' in normalized or 'convert' in normalized
        
        # Dangerous function detection
        dangerous_count = sum(1 for func in self.dangerous_functions if func in normalized)
        features['dangerous_function_count'] = dangerous_count
        features['has_xp_cmdshell'] = 'xp_cmdshell' in normalized or 'cmdshell' in normalized
        features['has_exec'] = 'exec' in normalized or 'execute' in normalized or 'sp_executesql' in normalized
        
        # Query complexity metrics
        features['query_complexity'] = self._calculate_query_complexity(normalized)
        features['nested_level'] = normalized.count('(')  # Simple nesting indicator
        
        # Additional operator detection
        features['greater_than_count'] = normalized.count('>')
        features['less_than_count'] = normalized.count('<')
        features['not_equal_count'] = normalized.count('!=') + normalized.count('<>')
        features['like_count'] = normalized.count('like')
        features['in_count'] = normalized.count(' in ') + normalized.count(' in(')
        
        return features
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score"""
        complexity = 0
        # Add points for various complexity indicators
        complexity += query.count('select') * 1
        complexity += query.count('join') * 2
        complexity += query.count('union') * 3
        complexity += query.count('(') * 0.5
        complexity += query.count('where') * 1
        complexity += query.count('and') * 0.5
        complexity += query.count('or') * 0.5
        return complexity
    
    def extract_pattern_features(self, query: str) -> Dict[str, Any]:
        """
        Extract pattern-based features for SQL injection detection.
        
        Args:
            query: SQL query string
            
        Returns:
            Dictionary of pattern features
        """
        features = {}
        normalized = self.preprocessor.normalize_query(query)
        
        # Check for injection patterns
        pattern_matches = 0
        for pattern in self.injection_patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                pattern_matches += 1
        
        features['injection_pattern_count'] = pattern_matches
        
        # Common injection techniques
        features['has_tautology'] = bool(re.search(r"(\d+\s*=\s*\d+|\'\d+\'\s*=\s*\'d+\')", normalized))
        features['has_union_attack'] = 'union' in normalized and 'select' in normalized
        features['has_comment_attack'] = '--' in normalized or '/*' in normalized
        features['has_boolean_attack'] = bool(re.search(r"(\bor\b|\band\b).*(\d+|\'|\")", normalized))
        features['has_time_delay'] = bool(re.search(r"(waitfor|delay|sleep|benchmark)", normalized, re.IGNORECASE))
        features['has_stacked_queries'] = normalized.count(';') > 1
        
        # Encoding patterns
        features['has_url_encoding'] = bool(re.search(r"(%27|%22|%3D|%3B|%2D)", normalized, re.IGNORECASE))
        features['has_hex_encoding'] = bool(re.search(r"0x[0-9a-f]+", normalized, re.IGNORECASE))
        
        # Suspicious character sequences
        features['has_suspicious_chars'] = bool(re.search(r"(\'|\"|;|--|\*|\/\*|\*\/)", normalized))
        
        # Additional injection techniques
        features['has_blind_injection'] = bool(re.search(r"(ascii|substring|char|length|len|count)", normalized, re.IGNORECASE))
        features['has_error_based'] = bool(re.search(r"(extractvalue|updatexml|exp|floor|rand)", normalized, re.IGNORECASE))
        features['has_second_order'] = normalized.count('select') > 1 and normalized.count('from') > 1
        features['has_piggybacked'] = normalized.count(';') > 0 and (normalized.count('select') > 1 or normalized.count('drop') > 0)
        
        # Encoding and obfuscation
        features['has_char_encoding'] = bool(re.search(r"(char\(|chr\(|ascii\()", normalized, re.IGNORECASE))
        features['has_unicode'] = bool(re.search(r"\\u[0-9a-f]{4}", normalized, re.IGNORECASE))
        features['has_double_encoding'] = normalized.count('%') > 2
        
        # SQL injection specific patterns
        features['has_always_true'] = bool(re.search(r"(\d+\s*=\s*\d+|'.*'\s*=\s*'.*'|1\s*=\s*1|'1'\s*=\s*'1')", normalized))
        features['has_always_false'] = bool(re.search(r"(\d+\s*=\s*0|'.*'\s*=\s*''|1\s*=\s*0|'1'\s*=\s*'0')", normalized))
        features['has_null_injection'] = bool(re.search(r"(null|nullif|isnull|coalesce)", normalized, re.IGNORECASE))
        
        # Advanced patterns
        features['has_union_all'] = 'union all' in normalized
        features['has_order_by_injection'] = bool(re.search(r"order\s+by\s+(\d+|'[^']*')", normalized, re.IGNORECASE))
        features['has_group_by_injection'] = 'group by' in normalized and bool(re.search(r"group\s+by\s+[^,]+,[^,]+", normalized, re.IGNORECASE))
        
        # Quote manipulation
        features['has_quote_escape'] = normalized.count("''") > 0 or normalized.count('""') > 0
        features['has_unclosed_quote'] = (normalized.count("'") % 2 != 0) or (normalized.count('"') % 2 != 0)
        
        # Command injection patterns
        features['has_command_injection'] = bool(re.search(r"(cmd|command|shell|system|exec|eval)", normalized, re.IGNORECASE))
        
        return features
    
    def extract_all_features(self, query: str) -> np.ndarray:
        """
        Extract all features from SQL query and return as numpy array.
        
        Args:
            query: SQL query string
            
        Returns:
            Numpy array of feature values
        """
        # Combine all feature types
        lexical = self.extract_lexical_features(query)
        syntactic = self.extract_syntactic_features(query)
        pattern = self.extract_pattern_features(query)
        
        # Combine all features
        all_features = {**lexical, **syntactic, **pattern}
        
        # Convert to array in consistent order
        feature_names = self.get_feature_names()
        
        feature_vector = np.array([all_features.get(name, 0) for name in feature_names])
        
        return feature_vector
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names in order.
        
        Returns:
            List of feature names
        """
        return [
            # Lexical features
            'length', 'word_count', 'char_count', 'digit_count', 'letter_count',
            'special_char_count', 'space_count', 'uppercase_count', 'lowercase_count',
            'digit_ratio', 'special_char_ratio', 'uppercase_ratio', 'letter_ratio', 'space_ratio',
            'entropy', 'unique_chars', 'most_common_char_freq', 'bigram_count',
            # Syntactic features
            'sql_keyword_count', 'has_select', 'has_union', 'has_or', 'has_and',
            'has_where', 'has_drop', 'has_delete', 'has_insert', 'has_update',
            'has_exec', 'equals_count', 'quote_count', 'semicolon_count',
            'comment_count', 'parentheses_count', 'has_comment', 'has_quotes',
            'has_semicolon', 'has_equals', 'sql_function_count', 'has_count',
            'has_concat', 'has_cast', 'dangerous_function_count', 'has_xp_cmdshell',
            'query_complexity', 'nested_level', 'greater_than_count', 'less_than_count',
            'not_equal_count', 'like_count', 'in_count',
            # Pattern features
            'injection_pattern_count', 'has_tautology', 'has_union_attack', 'has_comment_attack',
            'has_boolean_attack', 'has_time_delay', 'has_stacked_queries',
            'has_url_encoding', 'has_hex_encoding', 'has_suspicious_chars',
            'has_blind_injection', 'has_error_based', 'has_second_order', 'has_piggybacked',
            'has_char_encoding', 'has_unicode', 'has_double_encoding',
            'has_always_true', 'has_always_false', 'has_null_injection',
            'has_union_all', 'has_order_by_injection', 'has_group_by_injection',
            'has_quote_escape', 'has_unclosed_quote', 'has_command_injection'
        ]

