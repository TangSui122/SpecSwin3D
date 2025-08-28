import torch
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, CHECKPOINT_DIR

class BandImportanceCalculator:
    """Band importance calculator"""
    
    def __init__(self, input_dir, label_dir, sample_size=1000):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.sample_size = sample_size
        
        # Get file lists
        self.input_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "*.pt")))
        
        # Limit sample size
        if len(self.input_files) > sample_size:
            indices = np.random.RandomState(42).choice(len(self.input_files), sample_size, replace=False)
            self.input_files = [self.input_files[i] for i in indices]
            self.label_files = [self.label_files[i] for i in indices]
        
        print(f"Importance calculator initialized with {len(self.input_files)} samples")
    
    def load_data_samples(self):
        """Load data samples"""
        print("Loading data samples...")
        
        input_samples = []
        label_samples = []
        
        for i, (input_file, label_file) in enumerate(tqdm(zip(self.input_files, self.label_files), 
                                                          desc="Loading data", total=len(self.input_files))):
            try:
                # Load input data (16 bands)
                input_data = torch.load(input_file, map_location='cpu')
                input_tensor = input_data['input'] if isinstance(input_data, dict) else input_data
                
                # Load label data (219 bands)
                label_data = torch.load(label_file, map_location='cpu')
                label_tensor = label_data['label'] if isinstance(label_data, dict) else label_data
                
                # Flatten and convert to numpy
                input_flat = input_tensor.flatten().numpy()
                label_flat = label_tensor.flatten().numpy()  # All 219 bands flattened
                
                input_samples.append(input_flat)
                label_samples.append(label_flat)
                
            except Exception as e:
                print(f"Skipping file {input_file}: {e}")
                continue
        
        print(f"Successfully loaded {len(input_samples)} samples")
        return np.array(input_samples), np.array(label_samples)
    
    def calculate_variance_importance(self, label_samples):
        """Method 1: Variance-based importance"""
        print("Calculating variance importance...")
        
        # Reorganize data: all pixel values for each band
        num_bands = 219
        band_variances = []
        
        for band_idx in tqdm(range(num_bands), desc="Calculating variance"):
            # Extract all pixel values for this band
            band_pixels = []
            for sample in label_samples:
                # Each sample is flattened version of 128*128*219
                pixels_per_band = 128 * 128
                start_idx = band_idx * pixels_per_band
                end_idx = (band_idx + 1) * pixels_per_band
                band_pixels.extend(sample[start_idx:end_idx])
            
            variance = np.var(band_pixels)
            band_variances.append((band_idx, variance))
        
        # Sort by variance
        band_variances.sort(key=lambda x: x[1], reverse=True)
        
        print("Variance importance calculation completed")
        return band_variances
    
    def calculate_correlation_importance(self, input_samples, label_samples):
        """Method 2: Correlation-based importance"""
        print("Calculating correlation importance...")
        
        num_bands = 219
        correlation_scores = []
        
        # Calculate average correlation between each target band and 16 input bands
        for band_idx in tqdm(range(num_bands), desc="Calculating correlation"):
            band_correlations = []
            
            # Extract target band pixel values
            target_pixels = []
            for sample in label_samples:
                pixels_per_band = 128 * 128
                start_idx = band_idx * pixels_per_band
                end_idx = (band_idx + 1) * pixels_per_band
                target_pixels.extend(sample[start_idx:end_idx])
            
            target_pixels = np.array(target_pixels)
            
            # Calculate correlation with each input band
            for input_band in range(16):
                input_pixels = []
                for sample in input_samples:
                    pixels_per_band = 128 * 128
                    start_idx = input_band * pixels_per_band
                    end_idx = (input_band + 1) * pixels_per_band
                    input_pixels.extend(sample[start_idx:end_idx])
                
                input_pixels = np.array(input_pixels)
                
                # Calculate correlation coefficient
                if len(input_pixels) > 0 and len(target_pixels) > 0:
                    corr, _ = pearsonr(input_pixels, target_pixels)
                    if not np.isnan(corr):
                        band_correlations.append(abs(corr))
            
            # Average correlation as band importance
            avg_correlation = np.mean(band_correlations) if band_correlations else 0
            correlation_scores.append((band_idx, avg_correlation))
        
        # Sort by correlation
        correlation_scores.sort(key=lambda x: x[1], reverse=True)

        print("Correlation importance calculation completed")
        return correlation_scores
    
    def calculate_mutual_information_importance(self, input_samples, label_samples):
        """Method 3: Mutual information-based importance"""
        print("Calculating mutual information importance...")
        
        num_bands = 219
        mi_scores = []
        
        # Simplify: use mean of each sample to represent the sample
        input_means = np.mean(input_samples.reshape(len(input_samples), 16, -1), axis=2)  # (samples, 16)
        
        for band_idx in tqdm(range(num_bands), desc="Calculating mutual info"):
            # Extract target band means
            target_means = []
            for sample in label_samples:
                pixels_per_band = 128 * 128
                start_idx = band_idx * pixels_per_band
                end_idx = (band_idx + 1) * pixels_per_band
                band_mean = np.mean(sample[start_idx:end_idx])
                target_means.append(band_mean)
            
            target_means = np.array(target_means)
            
            # Calculate mutual information with input features
            mi_values = []
            for input_feature in range(16):
                mi = mutual_info_regression(input_means[:, input_feature:input_feature+1], target_means)
                mi_values.append(mi[0])
            
            # Average mutual information as importance
            avg_mi = np.mean(mi_values)
            mi_scores.append((band_idx, avg_mi))
        
        # Sort by mutual information
        mi_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("Mutual information importance calculation completed")
        return mi_scores
    
    def calculate_spectral_importance(self):
        """Method 4: Spectral physics-based importance"""
        print("Calculating spectral physics importance...")
        
        # Define importance weights for spectral regions
        spectral_weights = {
            # Visible region (0-79)
            **{i: 0.9 for i in range(0, 80)},
            # Near-infrared region (80-159) - important for vegetation analysis
            **{i: 1.0 for i in range(80, 160)},  
            # Short-wave infrared region (160-218) - important for mineral identification
            **{i: 0.8 for i in range(160, 219)}
        }
        
        # Additional weighting for key bands
        key_bands = {
            # Red edge region
            **{i: 1.2 for i in range(70, 85)},
            # Water vapor absorption bands
            **{i: 0.6 for i in [94, 114, 124, 164, 184]},
            # Vegetation characteristic bands
            **{i: 1.1 for i in [54, 68, 81, 103]}
        }
        
        # Update weights
        for band, weight in key_bands.items():
            if band in spectral_weights:
                spectral_weights[band] *= weight
        
        # Convert to list and sort
        spectral_scores = [(band, weight) for band, weight in spectral_weights.items()]
        spectral_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("Spectral physics importance calculation completed")
        return spectral_scores
    
    def design_importance_cascade_strategy(self, importance_scores, strategy_name):
        """Design cascade strategy based on importance scores"""
        print(f"Designing {strategy_name} cascade strategy...")
        
        # Select top 29 most important bands
        top_29_bands = [band_idx for band_idx, _ in importance_scores[:29]]
        
        # Design cascade strategy
        cascade_strategy = {
            0: top_29_bands[0:9],    # Most important 9 bands
            1: top_29_bands[9:12],   # Next important 3 bands
            2: top_29_bands[12:19],  # Third important 7 bands
            3: top_29_bands[19:22],  # Fourth important 3 bands
            4: top_29_bands[22:29]   # Relatively important 7 bands
        }
        
        print(f"{strategy_name} cascade strategy design completed")
        print(f"   Level 0: {cascade_strategy[0]}")
        print(f"   Level 1: {cascade_strategy[1]}")
        print(f"   Level 2: {cascade_strategy[2]}")
        print(f"   Level 3: {cascade_strategy[3]}")
        print(f"   Level 4: {cascade_strategy[4]}")
        
        return cascade_strategy
    
    def create_strategy_txt_files(self, results, output_dir):
        """Create strategy txt files for training"""
        print("Generating strategy txt files...")
        
        strategies_dir = os.path.join(output_dir, "strategies")
        os.makedirs(strategies_dir, exist_ok=True)
        
        strategy_files = {}
        
        for method, scores in results.items():
            # Design cascade strategy
            cascade_strategy = self.design_importance_cascade_strategy(scores, method)
            
            # Create txt file
            strategy_file = os.path.join(strategies_dir, f"{method}_strategy.txt")
            
            with open(strategy_file, 'w', encoding='utf-8') as f:
                f.write(f"# {method} importance strategy\n")
                f.write(f"# Generated time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"strategy_name: {method}_importance\n\n")
                
                for level in range(5):
                    bands = cascade_strategy[level]
                    bands_str = ", ".join(map(str, bands))
                    f.write(f"level_{level}: {bands_str}\n")
            
            strategy_files[method] = strategy_file
            print(f"{method} strategy file: {strategy_file}")
        
        return strategy_files
    
    def save_importance_results(self, results, output_dir):
        """Save importance calculation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(output_dir, "band_importance_analysis.json")
        
        # Convert to JSON serializable format
        json_results = {}
        for method, scores in results.items():
            json_results[method] = {
                'scores': [(int(band), float(score)) for band, score in scores],
                'top_10': [(int(band), float(score)) for band, score in scores[:10]],
                'cascade_strategy': self.design_importance_cascade_strategy(scores, method)
            }
        
        # Add metadata
        json_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(self.input_files),
            'total_bands': 219,
            'selected_bands': 29
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        print(f"Importance analysis results saved to: {results_file}")

        # Generate strategy txt files
        strategy_files = self.create_strategy_txt_files(results, output_dir)
        
        # Generate visualization
        self.create_importance_plots(results, output_dir)
        
        return results_file, strategy_files
    
    def create_importance_plots(self, results, output_dir):
        """Create importance analysis charts"""
        print("Generating importance analysis charts...")
        
        # Use compatible style settings
        try:
            plt.style.use('seaborn')  # Try general seaborn style
        except OSError:
            try:
                plt.style.use('ggplot')  # Try ggplot style
            except OSError:
                plt.style.use('default')  # Use default style
                print("⚠️ Using default style because seaborn style is unavailable")
    
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Band Importance Analysis', fontsize=16)
        
        methods = list(results.keys())
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (method, scores) in enumerate(results.items()):
            ax = axes[i//2, i%2]
            
            bands = [band for band, _ in scores[:50]]  # Show top 50
            importance_values = [score for _, score in scores[:50]]
            
            ax.plot(bands, importance_values, color=colors[i], linewidth=2, marker='o', markersize=3)
            ax.set_title(f'{method} Importance', fontsize=12)
            ax.set_xlabel('Band Index')
            ax.set_ylabel('Importance Score')
            ax.grid(True, alpha=0.3)
            
            # Mark top 10
            top_10_bands = [band for band, _ in scores[:10]]
            top_10_scores = [score for _, score in scores[:10]]
            ax.scatter(top_10_bands, top_10_scores, color='red', s=30, alpha=0.7, label='Top 10')
            ax.legend()
    
        plt.tight_layout()
        plot_file = os.path.join(output_dir, "importance_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Chart saved to: {plot_file}")
    
    def run_full_analysis(self):
        """Run complete importance analysis"""
        print("Starting complete band importance analysis...")
        
        # Load data
        input_samples, label_samples = self.load_data_samples()
        
        # Calculate importance using different methods
        results = {}
        
        # Method 1: Variance importance
        results['variance'] = self.calculate_variance_importance(label_samples)
        
        # Method 2: Correlation importance
        results['correlation'] = self.calculate_correlation_importance(input_samples, label_samples)
        
        # Method 3: Mutual information importance
        results['mutual_info'] = self.calculate_mutual_information_importance(input_samples, label_samples)
        
        # Method 4: Spectral physics importance
        results['spectral_physics'] = self.calculate_spectral_importance()
        
        # Save results
        output_dir = os.path.join(CHECKPOINT_DIR, "importance_analysis")
        results_file, strategy_files = self.save_importance_results(results, output_dir)
        
        print("Band importance analysis completed!")
        return results, results_file, strategy_files

def main():
    """Main function"""
    input_dir = os.path.join(BASE_DIR, "input_restacked_16")
    label_dir = os.path.join(BASE_DIR, "label")
    
    # Create importance calculator
    calculator = BandImportanceCalculator(input_dir, label_dir, sample_size=500)
    
    # Run analysis
    results, results_file, strategy_files = calculator.run_full_analysis()

    print(f"\nAnalysis completed!")
    print(f"Results file: {results_file}")
    print(f"Strategy files directory: {os.path.dirname(list(strategy_files.values())[0])}")

    print(f"\nTraining command examples:")
    for method, strategy_file in strategy_files.items():
        print(f"\n# {method}_importance strategy:")
        print(f"python train_SpecSwin3D_16.py --strategy custom --custom_txt {strategy_file} --batch_size 12")

if __name__ == "__main__":
    main()