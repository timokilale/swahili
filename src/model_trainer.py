# src/model_trainer.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import SwahiliFeatureExtractor

class SwahiliScamModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.feature_extractor = None
        
    def load_processed_data(self):
        """Load preprocessed features and labels"""
        print("📂 Loading processed data...")
        
        try:
            features = pd.read_csv('data/processed/features.csv')
            labels = pd.read_csv('data/processed/labels.csv')['label']
            
            print(f"✅ Loaded {len(features)} samples with {len(features.columns)} features")
            return features, labels
            
        except FileNotFoundError:
            print("❌ Processed data not found. Running feature extraction first...")
            from feature_extractor import load_and_process_data
            X_train, X_test, y_train, y_test, extractor = load_and_process_data()
            self.feature_extractor = extractor
            
            # Combine train and test for full dataset
            features = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
            labels = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
            
            return features, labels
    
    def train_models(self, X_train, y_train):
        """Train multiple ML models"""
        print("\n🤖 Training ML Models...")
        print("=" * 40)
        
        # Define models
        models_config = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"\n🔧 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            print(f"   Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Store model
            self.models[name] = {
                'model': model,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Track best model
            if cv_scores.mean() > self.best_score:
                self.best_score = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n🏆 Best model: {self.best_model_name} (CV Score: {self.best_score:.3f})")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n📊 MODEL EVALUATION")
        print("=" * 40)
        
        results = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Store results
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'CV Score': model_info['cv_score']
            })
            
            # Print results
            print(f"\n{name} Results:")
            print(f"  Accuracy:  {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1-Score:  {f1:.3f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        print(f"\n📈 SUMMARY TABLE:")
        print(results_df.round(3))
        
        return results_df
    
    def show_confusion_matrix(self, X_test, y_test):
        """Show confusion matrix for best model"""
        print(f"\n🎯 CONFUSION MATRIX - {self.best_model_name}")
        print("=" * 40)
        
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        print("Confusion Matrix:")
        print(f"                 Predicted")
        print(f"Actual    Safe  Scam")
        print(f"Safe      {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"Scam      {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nDetailed Metrics:")
        print(f"True Positives (Scams caught):     {tp}")
        print(f"True Negatives (Safe identified):  {tn}")
        print(f"False Positives (Safe flagged):    {fp}")
        print(f"False Negatives (Scams missed):    {fn}")
        
        if tp + fn > 0:
            scam_detection_rate = tp / (tp + fn)
            print(f"Scam Detection Rate: {scam_detection_rate:.1%}")
        
        if tn + fp > 0:
            safe_accuracy = tn / (tn + fp)
            print(f"Safe Message Accuracy: {safe_accuracy:.1%}")
    
    def show_feature_importance(self):
        """Show feature importance for best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            print(f"\n🔍 FEATURE IMPORTANCE - {self.best_model_name}")
            print("=" * 40)
            
            # Get feature importance
            importance = self.best_model.feature_importances_
            
            # Load feature names
            features = pd.read_csv('data/processed/features.csv')
            feature_names = features.columns
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Show top 10 features
            print("Top 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<25} | {row['importance']:.3f}")
    
    def save_best_model(self):
        """Save the best model and feature extractor"""
        print(f"\n💾 Saving best model ({self.best_model_name})...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, 'models/best_model.pkl')
        
        # Save feature extractor if available
        if self.feature_extractor:
            joblib.dump(self.feature_extractor, 'models/feature_extractor.pkl')
        
        # Save model info
        model_info = {
            'model_name': self.best_model_name,
            'accuracy': self.best_score,
            'features_count': len(pd.read_csv('data/processed/features.csv').columns),
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        pd.DataFrame([model_info]).to_csv('models/model_info.csv', index=False)
        
        print(f"✅ Model saved to: models/best_model.pkl")
        print(f"✅ Model info saved to: models/model_info.csv")
    
    def test_model_predictions(self):
        """Test model with sample messages"""
        print(f"\n🧪 TESTING MODEL PREDICTIONS")
        print("=" * 40)
        
        # Load feature extractor
        if not self.feature_extractor:
            try:
                self.feature_extractor = joblib.load('models/feature_extractor.pkl')
            except:
                print("⚠️ Feature extractor not found. Using new instance.")
                self.feature_extractor = SwahiliFeatureExtractor()
        
        # Test messages
        test_messages = [
            "Hongera! Umeshinda TSH 1,000,000. Piga *150*00# sasa kuconfirm.",
            "Habari za asubuhi. Mkutano wetu ni saa 2 jioni leo.",
            "URGENT! Tuma pesa haraka kwa namba hii 0712345678.",
            "Asante kwa huduma nzuri. Tuonane kesho asubuhi."
        ]
        
        expected = ["SCAM", "SAFE", "SCAM", "SAFE"]
        
        print("Testing sample messages:")
        for i, (message, exp) in enumerate(zip(test_messages, expected), 1):
            # Extract features for single message
            features = self.feature_extractor.extract_linguistic_features(message)
            
            # Convert to DataFrame with same columns as training data
            feature_df = pd.DataFrame([features])
            
            # Ensure all columns are present (fill missing with 0)
            training_features = pd.read_csv('data/processed/features.csv')
            # Get missing columns
            missing_cols = [col for col in training_features.columns if col not in feature_df.columns]

            # Add all missing columns at once using pd.concat
            if missing_cols:
                missing_df = pd.DataFrame(0, index=feature_df.index, columns=missing_cols)
                feature_df = pd.concat([feature_df, missing_df], axis=1)
            
            # Reorder columns to match training data
            feature_df = feature_df[training_features.columns]
            
            # Make prediction
            prediction = self.best_model.predict(feature_df)[0]
            confidence = self.best_model.predict_proba(feature_df)[0].max()
            
            result = "SCAM" if prediction == 1 else "SAFE"
            status = "✅" if result == exp else "❌"
            
            print(f"\n{i}. Message: {message[:50]}...")
            print(f"   Expected: {exp} | Predicted: {result} | Confidence: {confidence:.2f} {status}")

def main():
    """Main training pipeline"""
    print("🚀 SWAHILI SCAM DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Initialize trainer
    trainer = SwahiliScamModelTrainer()
    
    # Load data
    features, labels = trainer.load_processed_data()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    trainer.train_models(X_train, y_train)
    
    # Evaluate models
    results = trainer.evaluate_models(X_test, y_test)
    
    # Show detailed analysis
    trainer.show_confusion_matrix(X_test, y_test)
    trainer.show_feature_importance()
    
    # Save best model
    trainer.save_best_model()
    
    # Test predictions
    trainer.test_model_predictions()
    
    print(f"\n🎉 MODEL TRAINING COMPLETE!")
    print(f"🏆 Best Model: {trainer.best_model_name}")
    print(f"📊 Best Accuracy: {trainer.best_score:.1%}")
    print(f"💾 Model saved and ready for deployment!")

if __name__ == "__main__":
    main()