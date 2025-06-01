import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from scipy.stats import entropy
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

class ModelEnsemble:
    def __init__(self, csv_files: List[str], model_names: List[str] = None, 
                 model_weights: List[float] = None):
        """
        Initialize the ensemble system.
        
        Args:
            csv_files: List of paths to CSV files with predictions
            model_names: Optional list of model names (default: file basenames)
            model_weights: Optional weights for weighted ensemble (default: equal weights)
        """
        self.csv_files = csv_files
        self.model_names = model_names or [os.path.basename(f).split('.')[0] for f in csv_files]
        self.model_weights = model_weights or [1.0] * len(csv_files)
        
        # Normalize weights
        self.model_weights = np.array(self.model_weights)
        self.model_weights = self.model_weights / np.sum(self.model_weights)
        
        self.predictions = {}
        self.image_names = None
        self.class_names = None
        self.n_samples = None
        self.n_classes = None
        
    def load_predictions(self):
        """Load prediction data from CSV files."""
        print("Loading prediction files...")
        
        for i, (file_path, model_name) in enumerate(zip(self.csv_files, self.model_names)):
            df = pd.read_csv(file_path)
            
            # First row contains class names (excluding first column which is image names)
            if i == 0:
                self.class_names = df.columns[1:].tolist()
                self.image_names = df.iloc[:, 0].values
                self.n_samples = len(self.image_names)
                self.n_classes = len(self.class_names)
                print(f"Dataset: {self.n_samples} samples, {self.n_classes} classes")
                print(f"Classes: {self.class_names[:5]}{'...' if len(self.class_names) > 5 else ''}")
            
            # Extract probability predictions (skip first column with image names)
            prob_matrix = df.iloc[:, 1:].values
            
            # Ensure probabilities are valid (non-negative and sum to 1)
            prob_matrix = np.clip(prob_matrix, 1e-10, 1.0)  # Avoid zeros
            prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)  # Normalize
            
            self.predictions[model_name] = prob_matrix
            print(f"Loaded {model_name}: {prob_matrix.shape}")
    
    def average_ensemble(self) -> np.ndarray:
        """Simple average ensemble."""
        print("Computing average ensemble...")
        
        ensemble_preds = np.zeros((self.n_samples, self.n_classes))
        
        for model_name, weight in zip(self.model_names, self.model_weights):
            ensemble_preds += weight * self.predictions[model_name]
        
        return ensemble_preds
    
    def geometric_mean_ensemble(self) -> np.ndarray:
        """Geometric mean ensemble."""
        print("Computing geometric mean ensemble...")
        
        # Start with ones for geometric mean
        ensemble_preds = np.ones((self.n_samples, self.n_classes))
        
        for model_name, weight in zip(self.model_names, self.model_weights):
            preds = self.predictions[model_name]
            # Apply weight as exponent for geometric mean
            ensemble_preds *= np.power(preds, weight)
        
        # Normalize to ensure probabilities sum to 1
        ensemble_preds = ensemble_preds / ensemble_preds.sum(axis=1, keepdims=True)
        
        return ensemble_preds
    
    def max_ensemble(self) -> np.ndarray:
        """Max ensemble - take maximum probability for each class."""
        print("Computing max ensemble...")
        
        # Stack all predictions
        all_preds = np.stack([self.predictions[name] for name in self.model_names], axis=0)
        
        # Take maximum across models for each class
        ensemble_preds = np.max(all_preds, axis=0)
        
        # Normalize to ensure probabilities sum to 1
        ensemble_preds = ensemble_preds / ensemble_preds.sum(axis=1, keepdims=True)
        
        return ensemble_preds
    
    def majority_voting_ensemble(self) -> np.ndarray:
        """Majority voting ensemble based on predicted classes."""
        print("Computing majority voting ensemble...")
        
        ensemble_preds = np.zeros((self.n_samples, self.n_classes))
        
        for i in range(self.n_samples):
            # Get predicted classes from each model
            predicted_classes = []
            for model_name in self.model_names:
                pred_class = np.argmax(self.predictions[model_name][i])
                predicted_classes.append(pred_class)
            
            # Count votes for each class
            class_votes = np.bincount(predicted_classes, minlength=self.n_classes)
            
            # Convert votes to probabilities
            ensemble_preds[i] = class_votes / len(self.model_names)
        
        return ensemble_preds
    
    def weighted_rank_ensemble(self) -> np.ndarray:
        """Weighted rank-based ensemble."""
        print("Computing weighted rank ensemble...")
        
        ensemble_preds = np.zeros((self.n_samples, self.n_classes))
        
        for i in range(self.n_samples):
            sample_scores = np.zeros(self.n_classes)
            
            for model_name, weight in zip(self.model_names, self.model_weights):
                preds = self.predictions[model_name][i]
                
                # Get ranks (higher probability = lower rank number)
                ranks = np.argsort(np.argsort(-preds)) + 1
                
                # Convert ranks to scores (higher rank = lower score)
                scores = (self.n_classes + 1 - ranks) / self.n_classes
                
                sample_scores += weight * scores
            
            # Normalize scores to probabilities
            ensemble_preds[i] = sample_scores / np.sum(sample_scores)
        
        return ensemble_preds
    
    def temperature_scaling_ensemble(self, temperature: float = 1.0) -> np.ndarray:
        """Ensemble with temperature scaling."""
        print(f"Computing temperature scaled ensemble (T={temperature})...")
        
        ensemble_preds = np.zeros((self.n_samples, self.n_classes))
        
        for model_name, weight in zip(self.model_names, self.model_weights):
            preds = self.predictions[model_name]
            
            # Apply temperature scaling
            scaled_preds = softmax(np.log(preds + 1e-10) / temperature, axis=1)
            
            ensemble_preds += weight * scaled_preds
        
        return ensemble_preds
    
    def confidence_weighted_ensemble(self) -> np.ndarray:
        """Confidence-weighted ensemble - higher weight for more confident predictions."""
        print("Computing confidence-weighted ensemble...")
        
        ensemble_preds = np.zeros((self.n_samples, self.n_classes))
        
        for i in range(self.n_samples):
            sample_preds = np.zeros(self.n_classes)
            total_weight = 0
            
            for model_name in self.model_names:
                preds = self.predictions[model_name][i]
                
                # Calculate confidence as max probability
                confidence = np.max(preds)
                
                # Weight prediction by confidence
                sample_preds += confidence * preds
                total_weight += confidence
            
            # Normalize
            if total_weight > 0:
                ensemble_preds[i] = sample_preds / total_weight
            else:
                ensemble_preds[i] = sample_preds / len(self.model_names)
        
        return ensemble_preds
    
    def entropy_weighted_ensemble(self) -> np.ndarray:
        """Entropy-weighted ensemble - higher weight for less uncertain predictions."""
        print("Computing entropy-weighted ensemble...")
        
        ensemble_preds = np.zeros((self.n_samples, self.n_classes))
        
        for i in range(self.n_samples):
            sample_preds = np.zeros(self.n_classes)
            total_weight = 0
            
            for model_name in self.model_names:
                preds = self.predictions[model_name][i]
                
                # Calculate entropy (higher entropy = more uncertainty)
                pred_entropy = entropy(preds)
                
                # Weight is inverse of entropy (lower entropy = higher weight)
                weight = 1.0 / (1.0 + pred_entropy)
                
                sample_preds += weight * preds
                total_weight += weight
            
            # Normalize
            ensemble_preds[i] = sample_preds / total_weight
        
        return ensemble_preds
    
    def compute_all_ensembles(self, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """Compute all ensemble methods."""
        print("Computing all ensemble methods...")
        
        ensembles = {
            'average': self.average_ensemble(),
            'geometric_mean': self.geometric_mean_ensemble(),
            'max': self.max_ensemble(),
            'majority_voting': self.majority_voting_ensemble(),
            'weighted_rank': self.weighted_rank_ensemble(),
            'temperature_scaled': self.temperature_scaling_ensemble(temperature),
            'confidence_weighted': self.confidence_weighted_ensemble(),
            'entropy_weighted': self.entropy_weighted_ensemble()
        }
        
        return ensembles
    
    def save_ensemble_predictions(self, ensemble_preds: np.ndarray, 
                                output_path: str, ensemble_name: str = "ensemble"):
        """Save ensemble predictions to CSV file."""
        print(f"Saving {ensemble_name} predictions to {output_path}")
        
        # Create DataFrame
        df_data = {'ID': self.image_names}
        
        # Add class probabilities
        for i, class_name in enumerate(self.class_names):
            df_data[class_name] = ensemble_preds[:, i]
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        
        print(f"Saved {len(df)} predictions with {len(self.class_names)} classes")
    
    def evaluate_ensemble_performance(self, ensembles: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Evaluate ensemble performance metrics."""
        print("Evaluating ensemble performance...")
        
        results = []
        
        for method_name, preds in ensembles.items():
            # Calculate metrics
            max_probs = np.max(preds, axis=1)
            mean_confidence = np.mean(max_probs)
            
            # Entropy (uncertainty)
            pred_entropies = [entropy(pred) for pred in preds]
            mean_entropy = np.mean(pred_entropies)
            
            # Agreement with individual models
            agreements = []
            for model_name in self.model_names:
                model_preds = np.argmax(self.predictions[model_name], axis=1)
                ensemble_pred_classes = np.argmax(preds, axis=1)
                agreement = np.mean(model_preds == ensemble_pred_classes)
                agreements.append(agreement)
            
            mean_agreement = np.mean(agreements)
            
            # High confidence predictions (>0.9)
            high_conf_ratio = np.mean(max_probs > 0.9)
            
            results.append({
                'method': method_name,
                'mean_confidence': mean_confidence,
                'mean_entropy': mean_entropy,
                'mean_agreement_with_individuals': mean_agreement,
                'high_confidence_ratio': high_conf_ratio,
                'min_confidence': np.min(max_probs),
                'max_confidence': np.max(max_probs)
            })
        
        return pd.DataFrame(results)
    
    def find_ensemble_disagreements(self, ensembles: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Find samples where ensemble methods disagree."""
        print("Finding ensemble disagreements...")
        
        disagreements = []
        
        for i in range(self.n_samples):
            # Get predicted classes from each ensemble method
            ensemble_preds = {}
            for method, preds in ensembles.items():
                ensemble_preds[method] = np.argmax(preds[i])
            
            # Check if all ensemble methods agree
            pred_classes = list(ensemble_preds.values())
            if len(set(pred_classes)) > 1:  # Disagreement
                row_data = {
                    'ID': self.image_names[i],
                    'sample_idx': i
                }
                row_data.update(ensemble_preds)
                disagreements.append(row_data)
        
        return pd.DataFrame(disagreements)

def create_ensemble_predictions(csv_files: List[str], 
                              output_dir: str = "./ensemble_results",
                              model_names: List[str] = None,
                              model_weights: List[float] = None,
                              temperature: float = 1.0,
                              save_all_methods: bool = True) -> Dict:
    """
    Main function to create ensemble predictions from multiple CSV files.
    
    Args:
        csv_files: List of paths to CSV files
        output_dir: Directory to save ensemble results
        model_names: Optional model names
        model_weights: Optional model weights for weighted ensemble
        temperature: Temperature for temperature scaling
        save_all_methods: Whether to save all ensemble methods or just the best
        
    Returns:
        Dictionary containing ensemble results and analysis
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize ensemble
    ensemble = ModelEnsemble(csv_files, model_names, model_weights)
    ensemble.load_predictions()
    
    # Compute all ensemble methods
    ensembles = ensemble.compute_all_ensembles(temperature)
    
    # Evaluate performance
    performance_df = ensemble.evaluate_ensemble_performance(ensembles)
    performance_df = performance_df.sort_values('mean_confidence', ascending=False)
    
    print("\nEnsemble Performance Summary:")
    print(performance_df.round(4))
    
    # Save performance results
    performance_df.to_csv(os.path.join(output_dir, 'ensemble_performance.csv'), index=False)
    
    # Find disagreements between ensemble methods
    disagreements_df = ensemble.find_ensemble_disagreements(ensembles)
    if len(disagreements_df) > 0:
        disagreements_df.to_csv(os.path.join(output_dir, 'ensemble_disagreements.csv'), index=False)
        print(f"\nFound {len(disagreements_df)} samples with ensemble disagreements")
    
    # Save ensemble predictions
    if save_all_methods:
        for method_name, preds in ensembles.items():
            output_path = os.path.join(output_dir, f'{method_name}_ensemble_predictions.csv')
            ensemble.save_ensemble_predictions(preds, output_path, method_name)
    else:
        # Save only the best performing method (highest mean confidence)
        best_method = performance_df.iloc[0]['method']
        best_preds = ensembles[best_method]
        output_path = os.path.join(output_dir, f'best_ensemble_predictions.csv')
        ensemble.save_ensemble_predictions(best_preds, output_path, best_method)
        print(f"\nBest ensemble method: {best_method}")
    
    return {
        'ensemble_object': ensemble,
        'ensembles': ensembles,
        'performance': performance_df,
        'disagreements': disagreements_df
    }

# Example usage
if __name__ == "__main__":
    # Example configuration
    BASE = r"C:\Users\Acer\vscode"
    
    csv_files = [
        f'{BASE}/hvai_submissions/250528_baseline_0.3517.csv',
        f'{BASE}/hvai_submissions/250531_convnextbase_0.2501763867.csv',
        f'{BASE}/hvai_submissions/250531_v11x_0.2864.csv',
        f'{BASE}/hvai_submissions/250601_v11x_384.csv',
    ]
    model_names = ['ResNet18', 'ConvNext', 'YOLOv11', 'YOLOv11_384']
    
    # Optional: Set different weights for models
    # Higher weight = more influence in ensemble
    model_weights = [0.6, 1.0, 0.8, 0.6]  # Give EfficientNet more weight
    
    # Create ensemble predictions
    print("Starting ensemble prediction generation...")
    
    results = create_ensemble_predictions(
        csv_files=csv_files,
        output_dir='./ensemble_results',
        model_names=model_names,
        model_weights=model_weights,
        temperature=1.0,
        save_all_methods=True
    )
    
    print("\nEnsemble generation completed!")
    
    # Access results
    performance = results['performance']
    best_method = performance.iloc[0]['method']
    print(f"Recommended ensemble method: {best_method}")
    print(f"Mean confidence: {performance.iloc[0]['mean_confidence']:.4f}")
    
    # Example: Get predictions from best ensemble
    best_ensemble_preds = results['ensembles'][best_method]
    print(f"Best ensemble predictions shape: {best_ensemble_preds.shape}")