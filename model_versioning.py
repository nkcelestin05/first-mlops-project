import json
import pickle
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import hashlib


class ModelVersion:
    """Model versioning and metadata management."""
    
    def __init__(self, model_path='diabetes_model.pkl', metadata_path='model_metadata.json'):
        self.model_path = model_path
        self.metadata_path = metadata_path
    
    def compute_model_hash(self):
        """Compute SHA256 hash of the model file."""
        with open(self.model_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def save_metadata(self, X_test, y_test, model):
        """Save model metadata including version and metrics."""
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred))
        }
        
        # Create metadata
        metadata = {
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'created_at': datetime.now().isoformat(),
            'model_hash': self.compute_model_hash(),
            'model_type': type(model).__name__,
            'metrics': metrics,
            'training_samples': len(X_test),
            'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age']
        }
        
        # Load existing metadata
        try:
            with open(self.metadata_path, 'r') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = {'versions': []}
        
        # Add new version
        history['versions'].append(metadata)
        history['latest_version'] = metadata['version']
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✅ Model metadata saved: version {metadata['version']}")
        print(f"📊 Metrics: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
        
        return metadata
    
    def load_metadata(self):
        """Load model metadata."""
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'versions': [], 'latest_version': None}
    
    def get_latest_version(self):
        """Get the latest model version metadata."""
        metadata = self.load_metadata()
        if metadata['versions']:
            return metadata['versions'][-1]
        return None
    
    def compare_versions(self, version1_idx=-2, version2_idx=-1):
        """Compare two model versions."""
        metadata = self.load_metadata()
        if len(metadata['versions']) < 2:
            print("❌ Not enough versions to compare")
            return None
        
        v1 = metadata['versions'][version1_idx]
        v2 = metadata['versions'][version2_idx]
        
        print(f"\n📊 Comparing versions:")
        print(f"Version 1: {v1['version']} | Version 2: {v2['version']}")
        print(f"{'Metric':<15} {'V1':<10} {'V2':<10} {'Change':<10}")
        print("-" * 50)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            v1_val = v1['metrics'][metric]
            v2_val = v2['metrics'][metric]
            change = ((v2_val - v1_val) / v1_val * 100) if v1_val > 0 else 0
            print(f"{metric:<15} {v1_val:<10.3f} {v2_val:<10.3f} {change:>+9.2f}%")
        
        return {'v1': v1, 'v2': v2}
