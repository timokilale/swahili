# src/bilstm_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
import re
from collections import Counter
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from swahili_preprocessor import SwahiliPreprocessor

class SwahiliTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.preprocessor = SwahiliPreprocessor()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        sequence = self.text_to_sequence(text)
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def text_to_sequence(self, text):
        # Use proper Swahili preprocessing
        processed_text = self.preprocessor.preprocess(text)

        # Keep more characters for better vocabulary
        text = processed_text.lower()
        # Keep numbers and some punctuation that might be important for scam detection
        text = re.sub(r'[^\w\s*#]', ' ', text)
        words = text.split()

        sequence = []
        for word in words[:self.max_length]:
            if word in self.vocab:
                sequence.append(self.vocab[word])
            else:
                sequence.append(self.vocab['<UNK>'])

        # Pad sequence
        while len(sequence) < self.max_length:
            sequence.append(self.vocab['<PAD>'])

        return sequence[:self.max_length]

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()

        # Improved embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism for better feature extraction
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Dropout and classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim * 2]

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # [batch_size, seq_len, 1]
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_dim * 2]

        # Classification
        output = self.dropout(attended_output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)

        return output

class SwahiliBiLSTMTrainer:
    def __init__(self, data_path='data/raw/swahili_messages_sample.csv'):
        self.data_path = data_path
        self.vocab = {}
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {self.device}")
        
    def load_and_preprocess_data(self):
        print("📊 Loading dataset...")
        df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(df)} messages")
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        return texts, labels
    
    def build_vocabulary(self, texts, min_freq=1):
        print("🔤 Building vocabulary...")
        word_counts = Counter()
        preprocessor = SwahiliPreprocessor()

        for text in texts:
            # Use proper Swahili preprocessing
            processed_text = preprocessor.preprocess(text)
            text = processed_text.lower()
            # Keep numbers and important punctuation for scam detection
            text = re.sub(r'[^\w\s*#]', ' ', text)
            words = text.split()
            word_counts.update(words)

        # Build vocabulary with special tokens
        vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        idx = 4

        # Add words that appear at least min_freq times
        for word, count in word_counts.most_common():
            if count >= min_freq and len(word) > 1:  # Filter out single characters
                vocab[word] = idx
                idx += 1

        print(f"✅ Vocabulary size: {len(vocab)}")
        print(f"📊 Most common words: {list(word_counts.most_common(10))}")
        self.vocab = vocab
        return vocab
    
    def train_model(self, texts, labels, epochs=30, batch_size=16):
        print("🚀 Starting Bi-LSTM training...")

        # Build vocabulary
        self.build_vocabulary(texts)

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        print(f"📊 Training samples: {len(X_train)}")
        print(f"📊 Validation samples: {len(X_val)}")
        print(f"📊 Test samples: {len(X_test)}")

        # Create datasets
        train_dataset = SwahiliTextDataset(X_train, y_train, self.vocab)
        val_dataset = SwahiliTextDataset(X_val, y_val, self.vocab)
        test_dataset = SwahiliTextDataset(X_test, y_test, self.vocab)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        vocab_size = len(self.vocab)
        self.model = BiLSTMClassifier(vocab_size).to(self.device)

        # Loss and optimizer with weight decay
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

        best_val_acc = 0
        patience_counter = 0
        patience = 10

        print("\n🎯 TRAINING PROGRESS:")
        print("-" * 60)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_texts, batch_labels in train_loader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_texts)
                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            train_acc = 100 * correct / total

            # Validation phase
            val_acc, _, _, _ = self.evaluate_model(val_loader)

            # Learning rate scheduling
            scheduler.step(val_acc)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'vocab': self.vocab,
                    'epoch': epoch,
                    'val_acc': val_acc
                }, 'models/bilstm_best_model.pth')
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1:2d}/{epochs}] - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")

            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

        # Load best model for final evaluation
        checkpoint = torch.load('models/bilstm_best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Final evaluation
        test_accuracy, test_precision, test_recall, test_f1 = self.evaluate_model(test_loader)

        print("\n🎯 FINAL BI-LSTM RESULTS:")
        print("=" * 50)
        print(f"📊 Test Accuracy:  {test_accuracy:.2f}%")
        print(f"📊 Test Precision: {test_precision:.2f}%")
        print(f"📊 Test Recall:    {test_recall:.2f}%")
        print(f"📊 Test F1-Score:  {test_f1:.2f}%")
        print(f"📊 Best Val Acc:   {best_val_acc:.2f}%")

        self.save_model()

        return {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1
        }
    
    def evaluate_model(self, test_loader):
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch_texts, batch_labels in test_loader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_texts)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Handle edge cases for precision/recall calculation
        try:
            accuracy = accuracy_score(all_labels, all_predictions) * 100
            precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
            recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
            f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
        except Exception as e:
            print(f"⚠️ Warning in evaluation: {e}")
            accuracy = precision = recall = f1 = 0.0

        return accuracy, precision, recall, f1
    
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab
        }, 'models/bilstm_model.pth')
        
        with open('models/bilstm_vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)
        
        print("✅ Bi-LSTM model saved to models/bilstm_model.pth")

def main():
    print("🚀 SWAHILI BI-LSTM SCAM DETECTOR")
    print("=" * 50)

    try:
        trainer = SwahiliBiLSTMTrainer()
        texts, labels = trainer.load_and_preprocess_data()

        # Check data balance
        scam_count = sum(labels)
        total_count = len(labels)
        print(f"📊 Dataset balance: {scam_count} scam, {total_count - scam_count} legitimate")
        print(f"📊 Scam ratio: {scam_count/total_count:.2%}")

        # Train model
        results = trainer.train_model(texts, labels, epochs=30)

        print("\n🎉 BI-LSTM TRAINING COMPLETED!")
        print(f"🏆 Final Accuracy: {results['accuracy']:.2f}%")

        if results['accuracy'] >= 90:
            print("✅ EXCELLENT: Model achieved 90%+ accuracy!")
        elif results['accuracy'] >= 80:
            print("✅ GOOD: Model achieved 80%+ accuracy!")
        else:
            print("⚠️ Model needs improvement (accuracy < 80%)")

    except Exception as e:
        print(f"❌ Error in training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()