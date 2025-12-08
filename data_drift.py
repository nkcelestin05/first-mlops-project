import numpy as np
import json
from datetime import datetime
from scipy import stats


class DataDriftDetector:
    """Detect data drift in incoming predictions."""
    
    def __init__(self, reference_data_path='reference_data.json'):
        self.reference_data_path = reference_data_path
        self.reference_stats = self.load_reference_stats()
    
    def compute_statistics(self, data):
        """Compute statistical features of the data."""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75))
        }
    
    def save_reference_data(self, X, feature_names):
        """Save reference dataset statistics."""
        reference = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(X),
            'features': {}
        }
        
        for i, feature in enumerate(feature_names):
            reference['features'][feature] = self.compute_statistics(X[:, i])
        
        with open(self.reference_data_path, 'w') as f:
            json.dump(reference, f, indent=2)
        
        print(f"✅ Reference data saved with {len(X)} samples")
        self.reference_stats = reference
        return reference
    
    def load_reference_stats(self):
        """Load reference statistics."""
        try:
            with open(self.reference_data_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def ks_test(self, reference_data, current_data):
        """Perform Kolmogorov-Smirnov test for drift detection."""
        statistic, p_value = stats.ks_2samp(reference_data, current_data)
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'drift_detected': p_value < 0.05  # 5% significance level
        }
    
    def detect_drift(self, X_current, feature_names, threshold=0.05):
        """Detect data drift across all features."""
        if self.reference_stats is None:
            print("❌ No reference data found. Please save reference data first.")
            return None
        
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'features': {},
            'overall_drift': False
        }
        
        drifted_features = []
        
        for i, feature in enumerate(feature_names):
            current_stats = self.compute_statistics(X_current[:, i])
            reference = self.reference_stats['features'][feature]
            
            # Compare statistics
            mean_diff = abs(current_stats['mean'] - reference['mean']) / reference['std'] if reference['std'] > 0 else 0
            
            # Detect drift based on statistical distance
            drift_detected = mean_diff > 2  # 2 standard deviations
            
            if drift_detected:
                drifted_features.append(feature)
            
            drift_report['features'][feature] = {
                'current_stats': current_stats,
                'reference_stats': reference,
                'mean_difference_std': float(mean_diff),
                'drift_detected': drift_detected
            }
        
        drift_report['overall_drift'] = len(drifted_features) > 0
        drift_report['drifted_features'] = drifted_features
        
        return drift_report
    
    def print_drift_report(self, drift_report):
        """Print a formatted drift detection report."""
        if drift_report is None:
            return
        
        print("\n" + "="*60)
        print("📊 DATA DRIFT DETECTION REPORT")
        print("="*60)
        print(f"Timestamp: {drift_report['timestamp']}")
        print(f"Overall Drift Detected: {'🔴 YES' if drift_report['overall_drift'] else '🟢 NO'}")
        
        if drift_report['overall_drift']:
            print(f"Drifted Features: {', '.join(drift_report['drifted_features'])}")
        
        print("\nFeature-by-Feature Analysis:")
        print("-"*60)
        
        for feature, data in drift_report['features'].items():
            status = "🔴 DRIFT" if data['drift_detected'] else "🟢 OK"
            print(f"\n{feature} - {status}")
            print(f"  Current Mean: {data['current_stats']['mean']:.2f}")
            print(f"  Reference Mean: {data['reference_stats']['mean']:.2f}")
            print(f"  Difference (σ): {data['mean_difference_std']:.2f}")
        
        print("="*60)
