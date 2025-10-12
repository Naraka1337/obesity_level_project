#!/usr/bin/env python3
"""
Machine Learning Model Training Script for Obesity Prediction

This script trains multiple machine learning models to predict obesity levels
based on lifestyle factors and demographic information.

Usage:
    python train_models.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import joblib
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ObesityModelTrainer:
    """Main class for training obesity prediction models."""
    
    def __init__(self):
        """Initialize the trainer with paths and configurations."""
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.reports_dir = self.project_root / "reports"
        
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Data files
        self.processed_data_file = self.data_dir / "obesity_numeric.csv"
        self.mappings_file = self.data_dir / "mappings.json"
        
        # Model files
        self.scaler_file = self.models_dir / "scaler.pkl"
        self.best_model_file = self.models_dir / "best_model.pkl"
        self.all_models_file = self.models_dir / "all_models.pkl"
        
        # Report files
        self.training_report_file = self.reports_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Model configurations
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'SVM': SVC(random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Hyperparameter grids for tuning
        self.param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
        }
        
        self.results = {}
        
    def load_data(self):
        """Load and prepare the processed dataset."""
        logger.info("Loading processed dataset...")
        
        try:
            # Load processed data
            self.df = pd.read_csv(self.processed_data_file, sep=';')
            logger.info(f"Dataset loaded: {self.df.shape}")
            
            # Load mappings for reference
            with open(self.mappings_file, 'r') as f:
                self.mappings = json.load(f)
            
            # Separate features and target
            self.target = 'Obesity'
            self.features = [col for col in self.df.columns if col not in [self.target, 'BMI']]
            
            self.X = self.df[self.features]
            self.y = self.df[self.target]
            
            logger.info(f"Features: {len(self.features)}")
            logger.info(f"Samples: {len(self.X)}")
            logger.info(f"Target distribution:\n{self.y.value_counts().sort_index()}")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            logger.error("Please run preprocessing first: python src/preprocess.py")
            return False
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def split_data(self):
        """Split data into train, validation, and test sets."""
        logger.info("Splitting data into train/validation/test sets...")
        
        # First split: train+val vs test (80/20)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Second split: train vs val (75/25 of remaining 80%)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Train set: {self.X_train.shape[0]} samples")
        logger.info(f"Validation set: {self.X_val.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")
        
        # Scale features
        logger.info("Scaling features...")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Save scaler
        joblib.dump(self.scaler, self.scaler_file)
        logger.info(f"Scaler saved to: {self.scaler_file}")
    
    def train_models(self):
        """Train all models with hyperparameter tuning."""
        logger.info("Training models with hyperparameter tuning...")
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Use scaled data for SVM and Logistic Regression
                if model_name in ['SVM', 'Logistic Regression']:
                    X_train_use = self.X_train_scaled
                    X_val_use = self.X_val_scaled
                else:
                    X_train_use = self.X_train
                    X_val_use = self.X_val
                
                # Grid search for hyperparameter tuning
                grid_search = GridSearchCV(
                    model, self.param_grids[model_name],
                    cv=5, scoring='accuracy', n_jobs=-1, verbose=0
                )
                
                grid_search.fit(X_train_use, self.y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Train on full training set
                best_model.fit(X_train_use, self.y_train)
                
                # Evaluate on validation set
                val_predictions = best_model.predict(X_val_use)
                val_accuracy = accuracy_score(self.y_val, val_predictions)
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X_train_use, self.y_train, cv=5)
                
                # Store results
                self.results[model_name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'val_accuracy': val_accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'needs_scaling': model_name in ['SVM', 'Logistic Regression']
                }
                
                logger.info(f"{model_name} - Validation Accuracy: {val_accuracy:.4f}")
                logger.info(f"{model_name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                logger.info(f"{model_name} - Best Params: {grid_search.best_params_}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
    
    def evaluate_models(self):
        """Evaluate all models on test set."""
        logger.info("Evaluating models on test set...")
        
        for model_name, result in self.results.items():
            logger.info(f"Evaluating {model_name}...")
            
            try:
                model = result['model']
                
                # Use appropriate data (scaled or not)
                if result['needs_scaling']:
                    X_test_use = self.X_test_scaled
                else:
                    X_test_use = self.X_test
                
                # Predictions
                y_pred = model.predict(X_test_use)
                y_pred_proba = model.predict_proba(X_test_use) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                # Store test results
                result.update({
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'test_predictions': y_pred.tolist(),
                    'test_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
                })
                
                logger.info(f"{model_name} Test Results:")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
    
    def select_best_model(self):
        """Select the best model based on test accuracy."""
        logger.info("Selecting best model...")
        
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_accuracy'])
        best_model = self.results[best_model_name]['model']
        
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Test accuracy: {self.results[best_model_name]['test_accuracy']:.4f}")
        
        # Save best model
        joblib.dump(best_model, self.best_model_file)
        logger.info(f"Best model saved to: {self.best_model_file}")
        
        # Save all models
        joblib.dump(self.results, self.all_models_file)
        logger.info(f"All models saved to: {self.all_models_file}")
        
        return best_model_name, best_model
    
    def generate_report(self):
        """Generate comprehensive training report."""
        logger.info("Generating training report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(self.X),
                'features': self.features,
                'target_classes': sorted(self.y.unique().tolist()),
                'train_samples': len(self.X_train),
                'val_samples': len(self.X_val),
                'test_samples': len(self.X_test)
            },
            'models': {}
        }
        
        # Add model results
        for model_name, result in self.results.items():
            report['models'][model_name] = {
                'best_params': result['best_params'],
                'validation_accuracy': result['val_accuracy'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'test_accuracy': result['test_accuracy'],
                'test_precision': result['test_precision'],
                'test_recall': result['test_recall'],
                'test_f1': result['test_f1']
            }
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_accuracy'])
        report['best_model'] = best_model_name
        report['best_accuracy'] = self.results[best_model_name]['test_accuracy']
        
        # Save report
        with open(self.training_report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to: {self.training_report_file}")
        return report
    
    def print_summary(self):
        """Print training summary."""
        logger.info("="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        
        for model_name, result in self.results.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  Validation Accuracy: {result['val_accuracy']:.4f}")
            logger.info(f"  Test Accuracy: {result['test_accuracy']:.4f}")
            logger.info(f"  Test F1-Score: {result['test_f1']:.4f}")
        
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_accuracy'])
        logger.info(f"\n🏆 Best Model: {best_model_name}")
        logger.info(f"   Test Accuracy: {self.results[best_model_name]['test_accuracy']:.4f}")
        
        logger.info("\n✅ Training completed successfully!")
    
    def run_training(self):
        """Run the complete training pipeline."""
        logger.info("Starting obesity prediction model training...")
        
        # Load data
        if not self.load_data():
            return False
        
        # Split data
        self.split_data()
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Select best model
        self.select_best_model()
        
        # Generate report
        self.generate_report()
        
        # Print summary
        self.print_summary()
        
        return True

def main():
    """Main function."""
    trainer = ObesityModelTrainer()
    success = trainer.run_training()
    
    if success:
        logger.info("🎉 Model training pipeline completed successfully!")
    else:
        logger.error("❌ Model training pipeline failed!")
        exit(1)

if __name__ == "__main__":
    main()
