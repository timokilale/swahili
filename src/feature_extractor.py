# src/feature_extractor.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import sys
import os

# Add src directory to path to import our preprocessor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from swahili_preprocessor import SwahiliPreprocessor

class SwahiliFeatureExtractor:
    def __init__(self):
        self.preprocessor = SwahiliPreprocessor()
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.feature_names = []
        
    def extract_linguistic_features(self, text):
        """Extract custom Swahili linguistic features"""
        features = {}
        
        # Get basic features from preprocessor
        basic_features = self.preprocessor.extract_basic_features(text)
        features.update(basic_features)
        
        # Additional scam-specific features
        text_lower = text.lower()
        
        # Phone number patterns
        import re
        phone_pattern = r'0\d{9}|\+255\d{9}|\d{10}'
        features['phone_numbers'] = len(re.findall(phone_pattern, text))
        
        # Money amount patterns
        money_pattern = r'tsh\s*\d+|shilingi\s*\d+|\d+\s*tsh|\d+\s*shilingi'
        features['money_amounts'] = len(re.findall(money_pattern, text_lower))
        
        # Action words (typical in scams)
        action_words = ['piga', 'tuma', 'send', 'call', 'confirm', 'kuconfirm']
        features['action_words'] = sum(1 for word in action_words if word in text_lower)
        
        # Congratulatory words
        congrat_words = ['hongera', 'congratulations', 'bahati', 'ushindi', 'winner']
        features['congratulatory_words'] = sum(1 for word in congrat_words if word in text_lower)
        
        # Time pressure words
        time_words = ['sasa', 'haraka', 'urgent', 'now', 'immediately', 'papo hapo']
        features['time_pressure'] = sum(1 for word in time_words if word in text_lower)
        
        return features
    
    def extract_tfidf_features(self, texts):
        """Extract TF-IDF features from texts"""
        return self.tfidf.fit_transform(texts)
    
    def prepare_features(self, df):
        """Prepare all features for machine learning"""
        print("🔧 Extracting features from messages...")
        
        # Preprocess all texts
        df['cleaned_text'] = df['text'].apply(self.preprocessor.preprocess)
        
        # Extract linguistic features for each message
        linguistic_features = []
        for text in df['text']:
            features = self.extract_linguistic_features(text)
            linguistic_features.append(features)
        
        # Convert to DataFrame
        linguistic_df = pd.DataFrame(linguistic_features)
        
        # Extract TF-IDF features
        tfidf_features = self.extract_tfidf_features(df['cleaned_text'])
        
        # Get TF-IDF feature names
        tfidf_feature_names = [f'tfidf_{name}' for name in self.tfidf.get_feature_names_out()]
        
        # Convert TF-IDF to DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(), 
            columns=tfidf_feature_names,
            index=df.index
        )
        
        # Combine all features
        all_features = pd.concat([linguistic_df, tfidf_df], axis=1)
        
        # Store feature names
        self.feature_names = list(all_features.columns)
        
        print(f"✅ Extracted {len(self.feature_names)} features total")
        print(f"   - {len(linguistic_df.columns)} linguistic features")
        print(f"   - {len(tfidf_df.columns)} TF-IDF features")
        
        return all_features, df['label']
    
    def analyze_features(self, features, labels):
        """Analyze feature importance for scam detection"""
        print("\n📊 FEATURE ANALYSIS")
        print("=" * 40)
        
        # Calculate correlation with labels
        correlations = {}
        for col in features.columns:
            if features[col].dtype in ['int64', 'float64']:
                corr = features[col].corr(labels)
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
        
        # Sort by correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print("🔍 Top 10 Most Important Features:")
        for i, (feature, corr) in enumerate(sorted_features[:10], 1):
            print(f"{i:2d}. {feature:<25} | Correlation: {corr:.3f}")
        
        # Feature statistics
        print(f"\n📈 FEATURE STATISTICS:")
        print(f"Total features: {len(features.columns)}")
        print(f"Numeric features: {len([col for col in features.columns if features[col].dtype in ['int64', 'float64']])}")
        print(f"Features with correlation > 0.1: {len([f for f, c in correlations.items() if c > 0.1])}")
        
        return sorted_features

def load_and_process_data():
    """Load dataset and extract features"""
    print("📂 Loading dataset...")
    
    # Load the dataset
    df = pd.read_csv('data/raw/swahili_messages_sample.csv')
    print(f"✅ Loaded {len(df)} messages")
    
    # Initialize feature extractor
    extractor = SwahiliFeatureExtractor()
    
    # Extract features
    features, labels = extractor.prepare_features(df)
    
    # Analyze features
    feature_importance = extractor.analyze_features(features, labels)
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"\n📊 DATA SPLIT:")
    print(f"Training set: {len(X_train)} messages")
    print(f"Test set: {len(X_test)} messages")
    print(f"Training scam ratio: {y_train.mean():.1%}")
    print(f"Test scam ratio: {y_test.mean():.1%}")
    
    # Save processed features
    features.to_csv('data/processed/features.csv', index=False)
    labels.to_csv('data/processed/labels.csv', index=False)
    
    print(f"\n💾 Features saved to: data/processed/features.csv")
    print(f"💾 Labels saved to: data/processed/labels.csv")
    
    return X_train, X_test, y_train, y_test, extractor

if __name__ == "__main__":
    print("🚀 SWAHILI FEATURE EXTRACTION")
    print("=" * 50)
    
    try:
        X_train, X_test, y_train, y_test, extractor = load_and_process_data()
        
        print("\n✅ Feature extraction completed successfully!")
        print("🎯 Ready for model training!")
        
        # Show sample of extracted features
        print("\n📋 SAMPLE FEATURES (first 5 rows):")
        print(X_train.head())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the dataset file exists in data/raw/")