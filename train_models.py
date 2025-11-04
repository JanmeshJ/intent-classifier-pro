"""
Multi-Model Training Pipeline
Trains 4 different models: Logistic Regression, Random Forest, SVM, XGBoost
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not available (optional)")

class MultiModelTrainer:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.label_map = {}
        self.results = {}
        
    def load_data(self, filepath='data/customer_intents.csv'):
        print(f"\n📂 Loading dataset: {filepath}")
        df = pd.read_csv(filepath)
        print(f"   ✓ Loaded {len(df)} samples")
        print(f"   ✓ Intent classes: {df['intent'].nunique()}")
        return df
    
    def prepare_data(self, df):
        X = df['text'].values
        y = df['intent'].values
        
        # Create label mapping
        unique_labels = sorted(df['intent'].unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_map = {idx: label for label, idx in self.label_map.items()}
        
        y_encoded = np.array([self.label_map[label] for label in y])
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\n✓ Training samples: {len(X_train)}")
        print(f"✓ Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def vectorize(self, X_train, X_test):
        print("\n📊 Vectorizing text...")
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        return X_train_vec, X_test_vec
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        models_config = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        if HAS_XGBOOST:
            models_config['xgboost'] = XGBClassifier(
                random_state=42, 
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        
        for name, model in models_config.items():
            print(f"\n🔄 Training {name.upper()}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            self.models[name] = model
            self.results[name] = {
                'accuracy': float(accuracy),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            print(f"   ✓ Accuracy: {accuracy:.4f}")
            print(f"   ✓ CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Print comparison
        self.print_comparison()
    
    def print_comparison(self):
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(f"\n{'Model':<25} {'Accuracy':<12} {'CV Score':<12}")
        print("-" * 70)
        
        for name, metrics in sorted(self.results.items(), 
                                   key=lambda x: x[1]['accuracy'], 
                                   reverse=True):
            print(f"{name:<25} {metrics['accuracy']:<12.4f} {metrics['cv_mean']:<12.4f}")
        
        best = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print("\n" + "=" * 70)
        print(f"🏆 BEST MODEL: {best[0].upper()} (Accuracy: {best[1]['accuracy']:.4f})")
        print("=" * 70)
    
    def save_models(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = Path('models') / timestamp
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            with open(version_dir / f'{name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save vectorizer
        with open(version_dir / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save label mapping
        with open(version_dir / 'label_map.json', 'w') as f:
            json.dump(self.label_map, f)
        
        # Save results
        with open(version_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Update current pointer (Windows-compatible)
        current_file = Path('models') / 'current'
        with open(current_file, 'w') as f:
            f.write(timestamp)
        
        print(f"\n✓ Models saved: {timestamp}")
        print(f"   Path: {version_dir}")
        return timestamp

def main():
    print("=" * 70)
    print("MULTI-MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    trainer = MultiModelTrainer()
    
    # Load data
    df = trainer.load_data()
    
    # Prepare
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Vectorize
    X_train_vec, X_test_vec = trainer.vectorize(X_train, X_test)
    
    # Train all models
    trainer.train_all_models(X_train_vec, X_test_vec, y_train, y_test)
    
    # Save
    trainer.save_models()
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
