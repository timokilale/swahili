# src/bert_model.py
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle

class SwahiliBERTDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SwahiliBERTTrainer:
    def __init__(self, data_path='data/raw/swahili_messages_sample.csv'):
        self.data_path = data_path
        self.tokenizer = None
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
    
    def initialize_model(self):
        print("🤖 Initializing multilingual BERT...")
        model_name = "bert-base-multilingual-cased"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        
        print("✅ BERT model initialized")
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        accuracy = accuracy_score(labels, predictions)
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, texts, labels):
        print("🚀 Starting BERT fine-tuning...")
        
        self.initialize_model()
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_dataset = SwahiliBERTDataset(X_train, y_train, self.tokenizer)
        test_dataset = SwahiliBERTDataset(X_test, y_test, self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir='./models/bert_results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./models/bert_logs',
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        
        # Final evaluation
        results = trainer.evaluate()
        
        print("\n🎯 FINAL BERT RESULTS:")
        print("=" * 50)
        print(f"📊 Test Accuracy:  {results['eval_accuracy']*100:.2f}%")
        print(f"📊 Test Precision: {results['eval_precision']*100:.2f}%")
        print(f"📊 Test Recall:    {results['eval_recall']*100:.2f}%")
        print(f"📊 Test F1-Score:  {results['eval_f1']*100:.2f}%")
        
        self.save_model()
        
        return {
            'accuracy': results['eval_accuracy']*100,
            'precision': results['eval_precision']*100,
            'recall': results['eval_recall']*100,
            'f1_score': results['eval_f1']*100
        }
    
    def save_model(self):
        self.model.save_pretrained('models/bert_model')
        self.tokenizer.save_pretrained('models/bert_tokenizer')
        print("✅ BERT model saved to models/bert_model")

def main():
    print("🚀 SWAHILI BERT SCAM DETECTOR")
    print("=" * 50)
    
    trainer = SwahiliBERTTrainer()
    texts, labels = trainer.load_and_preprocess_data()
    results = trainer.train_model(texts, labels)

if __name__ == "__main__":
    main()