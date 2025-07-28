# src/swahili_preprocessor.py
import re
import nltk
from collections import Counter

class SwahiliPreprocessor:
    def __init__(self):
        # Swahili stopwords
        self.swahili_stopwords = [
            'na', 'ya', 'wa', 'kwa', 'ni', 'si', 'la', 'za', 
            'cha', 'vya', 'ma', 'pa', 'ku', 'mu', 'ki', 'vi',
            'au', 'kama', 'lakini', 'pia', 'tu', 'hata', 'bado',
            'hii', 'hiyo', 'hilo', 'hayo', 'hao', 'wale', 'ile',
            'katika', 'kwenye', 'juu', 'chini', 'mbele', 'nyuma'
        ]
        
        # Swahili slang normalization dictionary
        self.slang_dict = {
            'pesa': ['pesa', 'doo', 'hela', 'fedha', 'cash', 'kash'],
            'haraka': ['haraka', 'upesi', 'hima', 'speed', 'haraka haraka'],
            'ushindi': ['ushindi', 'kushinda', 'win', 'rbh', 'shinda'],
            'bure': ['bure', 'free', 'bila malipo', 'bila gharama'],
            'zawadi': ['zawadi', 'gift', 'tuzo', 'prize', 'giveaway'],
            'simu': ['simu', 'phone', 'foni', 'rununu'],
            'sasa': ['sasa', 'now', 'mara moja', 'papo hapo']
        }
        
        # Scam-related keywords for detection
        self.scam_keywords = [
            'pesa', 'ushindi', 'haraka', 'bure', 'zawadi', 
            'malipo', 'fedha', 'shilingi', 'dola', 'tsh',
            'bahati', 'mchezo', 'lottery', 'jackpot'
        ]
        
        # Urgency indicators
        self.urgency_words = [
            'haraka', 'sasa', 'mara moja', 'upesi', 'urgent',
            'papo hapo', 'hima', 'speed', 'mapema'
        ]
        
        # Money-related terms
        self.money_terms = [
            'shilingi', 'dola', 'malipo', 'fedha', 'pesa', 'tsh',
            'elfu', 'milioni', 'laki', 'cash', 'hela', 'doo'
        ]
        
        # Emotional manipulation words
        self.emotion_words = [
            'familia', 'mama', 'baba', 'mtoto', 'harusi', 'mazishi',
            'ugonjwa', 'hospitali', 'msaada', 'huzuni', 'furaha'
        ]
    
    def clean_text(self, text):
        """Clean and normalize Swahili text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers (Tanzanian format)
        text = re.sub(r'\+?255\d{9}|\+?\d{10,}|0\d{9}', '', text)
        
        # Remove special characters but keep Swahili letters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def normalize_slang(self, text):
        """Normalize Swahili slang to standard terms"""
        for standard, variants in self.slang_dict.items():
            for variant in variants:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(variant) + r'\b'
                text = re.sub(pattern, standard, text, flags=re.IGNORECASE)
        return text
    
    def remove_stopwords(self, text):
        """Remove Swahili stopwords"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.swahili_stopwords]
        return ' '.join(filtered_words)
    
    def detect_language_mix(self, text):
        """Detect code-switching between Swahili and English"""
        english_words = ['money', 'cash', 'win', 'free', 'prize', 'urgent', 
                        'congratulations', 'winner', 'call', 'send', 'now']
        swahili_words = ['pesa', 'haraka', 'ushindi', 'bure', 'zawadi', 
                        'hongera', 'mshindi', 'piga', 'tuma', 'sasa']
        
        words = text.lower().split()
        eng_count = sum(1 for word in words if word in english_words)
        swa_count = sum(1 for word in words if word in swahili_words)
        total_words = len(words)
        
        return {
            'english_ratio': eng_count / max(total_words, 1),
            'swahili_ratio': swa_count / max(total_words, 1),
            'mixed_language': eng_count > 0 and swa_count > 0
        }
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        # Step 1: Clean text
        text = self.clean_text(text)
        
        # Step 2: Normalize slang
        text = self.normalize_slang(text)
        
        # Step 3: Remove stopwords
        text = self.remove_stopwords(text)
        
        return text
    
    def extract_basic_features(self, text):
        """Extract basic features for scam detection"""
        features = {}
        
        # Original text stats
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = sum(len(word) for word in text.split()) / max(len(text.split()), 1)
        
        # Scam keyword detection
        text_lower = text.lower()
        features['scam_keyword_count'] = sum(1 for keyword in self.scam_keywords if keyword in text_lower)
        
        # Urgency detection
        features['urgency_score'] = sum(1 for word in self.urgency_words if word in text_lower)
        
        # Money mentions
        features['money_mentions'] = sum(1 for term in self.money_terms if term in text_lower)
        
        # Emotional manipulation
        features['emotion_score'] = sum(1 for word in self.emotion_words if word in text_lower)
        
        # Punctuation patterns
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Language mixing
        lang_features = self.detect_language_mix(text)
        features.update(lang_features)
        
        return features

# Test the preprocessor
if __name__ == "__main__":
    preprocessor = SwahiliPreprocessor()
    
    # Test messages
    test_messages = [
        "Hongera! Umeshinda TSH 1,000,000. Piga *150*00# sasa kuconfirm.",
        "Habari za asubuhi. Mkutano wetu ni saa 2 jioni leo.",
        "URGENT! Mama yako amelazwa hospitalini. Tuma pesa haraka 0712345678."
    ]
    
    print("🧪 Testing Swahili Preprocessor...")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n📝 Test Message {i}:")
        print(f"Original: {message}")
        
        # Preprocess
        cleaned = preprocessor.preprocess(message)
        print(f"Cleaned:  {cleaned}")
        
        # Extract features
        features = preprocessor.extract_basic_features(message)
        print(f"Features: {features}")
        
        # Determine if likely scam
        scam_score = (features['scam_keyword_count'] + 
                     features['urgency_score'] + 
                     features['money_mentions'])
        
        if scam_score >= 2:
            print("🚨 LIKELY SCAM")
        else:
            print("✅ LIKELY SAFE")
        
        print("-" * 30)
    
    print("\n✅ Preprocessor test complete!")