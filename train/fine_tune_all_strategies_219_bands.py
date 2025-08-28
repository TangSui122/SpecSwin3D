import sys
import os
import json
import contextlib
import glob
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use relative paths that adapt to both local and server environments
def get_environment_paths():
    """Auto-detect runtime environment and return base/data/checkpoint paths."""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # train directory
    project_root = os.path.dirname(current_dir)  # project root
    
    # Linux/Unix vs Windows
    if os.name == 'posix':
        print("Detected Linux/Unix environment")
        base_dir = os.path.join(project_root, "dataset")
        checkpoint_dir = os.path.join(project_root, "checkpoints")
    else:
        print("Detected Windows environment")
        base_dir = os.path.join(project_root, "dataset")
        checkpoint_dir = os.path.join(project_root, "checkpoints")
    
    return base_dir, checkpoint_dir

# Resolve environment paths
BASE_DIR, CHECKPOINT_DIR = get_environment_paths()

print("Environment path settings:")
print(f"   BASE_DIR: {BASE_DIR}")
print(f"   CHECKPOINT_DIR: {CHECKPOINT_DIR}")
print(f"   CHECKPOINT_DIR exists: {os.path.exists(CHECKPOINT_DIR)}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse

# Import existing modules
from train.train_SpecSwin3D_16 import (
    SpecSwin_SingleBand, SingleBandDataset, 
    get_spectral_similarity, evaluate_model,
    denormalize_prediction, create_single_band_loaders
)

def test_checkpoint_access():
    """Test access to the checkpoint directory."""
    print("\nTesting checkpoint access")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"   Exists: {os.path.exists(CHECKPOINT_DIR)}")
    
    if os.path.exists(CHECKPOINT_DIR):
        try:
            items = os.listdir(CHECKPOINT_DIR)
            print(f"   Directory items ({len(items)}): {items}")
            
            # Try a common strategy folder
            test_strategy = "correlation_importance"
            test_path = os.path.join(CHECKPOINT_DIR, test_strategy)
            print(f"\nTesting strategy name: {test_strategy}")
            print(f"   Strategy path: {test_path}")
            print(f"   Strategy exists: {os.path.exists(test_path)}")
            
            if os.path.exists(test_path):
                strategy_items = os.listdir(test_path)
                print(f"   Strategy contents: {strategy_items}")
                
                models_dir = os.path.join(test_path, "models")
                print(f"   Models directory: {models_dir}")
                print(f"   Models directory exists: {os.path.exists(models_dir)}")
                
                if os.path.exists(models_dir):
                    model_files = os.listdir(models_dir)
                    print(f"   Model files count: {len(model_files)}")
                    if model_files:
                        print(f"   First 5 model files: {model_files[:5]}")
                        
                        # Test glob pattern
                        pattern = os.path.join(models_dir, "band_*.pth")
                        matched_files = glob.glob(pattern)
                        print(f"   Glob matches: {len(matched_files)}")
                else:
                    print("   Models directory does not exist")
            else:
                print("   Strategy directory does not exist")
        except Exception as e:
            print(f"   Failed to access directory: {e}")
    else:
        print("   Checkpoint directory does not exist")

def get_available_strategies(checkpoint_dir):
    """List all available strategies under the checkpoint directory."""
    print(f"\nInspecting checkpoint directory: {checkpoint_dir}")
    print(f"   Exists: {os.path.exists(checkpoint_dir)}")
    
    strategies = []
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return strategies
    
    try:
        # List subdirectories
        all_items = os.listdir(checkpoint_dir)
        print(f"Items under checkpoint ({len(all_items)}): {all_items}")
        
        exclude_dirs = ['backup_old_training', 'importance_analysis']
        
        for item in all_items:
            item_path = os.path.join(checkpoint_dir, item)
            
            if os.path.isdir(item_path) and item not in exclude_dirs:
                print(f"\nChecking strategy directory: {item}")
                
                models_dir = os.path.join(item_path, "models")
                print(f"   Models directory: {models_dir}")
                print(f"   Exists: {os.path.exists(models_dir)}")
                
                if os.path.exists(models_dir):
                    try:
                        all_files = os.listdir(models_dir)
                        pth_files = [f for f in all_files if f.endswith('.pth')]
                        band_files = [f for f in pth_files if f.startswith('band_')]
                        
                        print(f"   Total files: {len(all_files)}")
                        print(f"   .pth files: {len(pth_files)}")
                        print(f"   band_*.pth files: {len(band_files)}")
                        
                        if band_files:
                            print(f"   First 3 band files: {band_files[:3]}")
                            strategies.append((item, len(band_files)))
                            print(f"   Strategy {item} is available ({len(band_files)} models)")
                        else:
                            print(f"   Strategy {item} has no band_* .pth files")
                            if pth_files:
                                print(f"   Other .pth files: {pth_files[:3]}")
                    except Exception as e:
                        print(f"   Failed to inspect model files: {e}")
                else:
                    print(f"   Strategy {item} has no 'models' directory")
    except Exception as e:
        print(f"Failed to list checkpoint directory: {e}")
    
    print(f"\nFound {len(strategies)} available strategies")
    return sorted(strategies, key=lambda x: x[1], reverse=True)

def main():
    # CLI
    parser = argparse.ArgumentParser(description='Full-spectrum fine-tuning for 219 bands across strategies')
    parser.add_argument('--strategy', type=str, default=None,
                       help='Specify a single strategy to process')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Path to checkpoint directory')
    parser.add_argument('--base_dir', type=str, default=None,
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Training device, e.g., "cuda" or "cpu"')
    parser.add_argument('--start_band', type=int, default=0,
                       help='Start band index (inclusive)')
    parser.add_argument('--end_band', type=int, default=223,  
                       help='End band index (inclusive)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Plan-only mode: show plan without training')
    parser.add_argument('--run_actual', action='store_true',
                       help='Run actual fine-tuning instead of dry run')
    
    args = parser.parse_args()
    
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else CHECKPOINT_DIR
    base_dir = args.base_dir if args.base_dir else BASE_DIR
    
    print("Starting full-spectrum fine-tuning (219 bands)")
    print(f"Using checkpoint directory: {checkpoint_dir}")
    print(f"Using dataset directory: {base_dir}")
    
    # Test checkpoint access first
    test_checkpoint_access_with_path(checkpoint_dir)
    
    # Discover available strategies
    available_strategies = get_available_strategies(checkpoint_dir)
    
    if not available_strategies:
        print("\nNo available strategies were found")
        return
    
    print(f"\nDiscovered {len(available_strategies)} available strategies:")
    for i, (strategy, model_count) in enumerate(available_strategies, 1):
        print(f"   {i:2d}. {strategy:<30} ({model_count} models)")
    
    # Decide which strategies to process
    if args.strategy:
        if args.strategy not in [s[0] for s in available_strategies]:
            print(f"Specified strategy '{args.strategy}' does not exist")
            print(f"Available strategies: {[s[0] for s in available_strategies]}")
            return
        strategies_to_process = [args.strategy]
    else:
        # Process all strategies
        strategies_to_process = [s[0] for s in available_strategies]
    
    print(f"\nStrategies to process: {strategies_to_process}")
    
    # Process each strategy
    for strategy in strategies_to_process:
        print(f"\n{'='*80}")
        print(f"Processing strategy: {strategy}")
        print(f"{'='*80}")
        
        try:
            # Create fine-tuner with explicit checkpoint_dir
            fine_tuner = AllBandFineTuner(strategy, checkpoint_dir)
            
            # Run plan or actual fine-tuning
            plan = fine_tuner.run_full_spectrum_fine_tuning(
                device=args.device,
                batch_size=args.batch_size,
                start_band=args.start_band,
                end_band=args.end_band,
                dry_run=not args.run_actual
            )
            
            print(f"Strategy {strategy} finished successfully")
            
        except Exception as e:
            print(f"Strategy {strategy} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

def test_checkpoint_access_with_path(checkpoint_dir):
    """Test checkpoint access using the provided path."""
    print(f"\nTesting checkpoint access")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"   Exists: {os.path.exists(checkpoint_dir)}")
    
    if os.path.exists(checkpoint_dir):
        try:
            items = os.listdir(checkpoint_dir)
            print(f"   Directory items ({len(items)}): {items}")
            
            # Try a common strategy name
            test_strategy = "correlation_importance"
            test_path = os.path.join(checkpoint_dir, test_strategy)
            print(f"\nTesting strategy: {test_strategy}")
            print(f"   Path: {test_path}")
            print(f"   Exists: {os.path.exists(test_path)}")
            
            if os.path.exists(test_path):
                strategy_items = os.listdir(test_path)
                print(f"   Strategy contents: {strategy_items}")
                
                models_dir = os.path.join(test_path, "models")
                print(f"   Models directory: {models_dir}")
                print(f"   Models directory exists: {os.path.exists(models_dir)}")
                
                if os.path.exists(models_dir):
                    model_files = os.listdir(models_dir)
                    print(f"   Model files count: {len(model_files)}")
                    if model_files:
                        print(f"   First 5 model files: {model_files[:5]}")
                        
                        # Test glob
                        pattern = os.path.join(models_dir, "band_*.pth")
                        matched_files = glob.glob(pattern)
                        print(f"   Glob matches: {len(matched_files)}")
                else:
                    print("   Models directory does not exist")
            else:
                print("   Strategy directory does not exist")
        except Exception as e:
            print(f"   Failed to access directory: {e}")
    else:
        print("   Checkpoint directory does not exist")

class AllBandFineTuner:
    """Fine-tune models for the full spectrum (219 target bands)."""
    
    def __init__(self, source_strategy, checkpoint_dir):
        self.source_strategy = source_strategy
        self.checkpoint_dir = checkpoint_dir
        self.source_dir = os.path.join(checkpoint_dir, source_strategy)
        self.models_dir = os.path.join(self.source_dir, "models")
        
        print("Inspect source strategy directories:")
        print(f"   Source directory: {self.source_dir}")
        print(f"   Models directory: {self.models_dir}")
        print(f"   Source directory exists: {os.path.exists(self.source_dir)}")
        print(f"   Models directory exists: {os.path.exists(self.models_dir)}")
        
        # Output directories
        self.output_strategy = f"{source_strategy}_full_219_bands"
        self.output_dir = os.path.join(checkpoint_dir, self.output_strategy)
        self.output_models_dir = os.path.join(self.output_dir, "models")
        self.output_logs_dir = os.path.join(self.output_dir, "tensorboard_logs")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_models_dir, exist_ok=True)
        os.makedirs(self.output_logs_dir, exist_ok=True)
        
        print("Initialized full-spectrum fine-tuning")
        print(f"   Source strategy: {source_strategy}")
        print(f"   Source models directory: {self.models_dir}")
        print(f"   Output strategy: {self.output_strategy}")
        print(f"   Output directory: {self.output_dir}")
        
        # Load source strategy info
        self.load_source_strategy_info()
        
        # Configuration consistent with your training code
        self.input_bands = [30, 20, 9, 40, 52]   # 5 input bands
        self.total_bands = 224                   # 0-223
        self.target_model_count = 219            # 224 - 5 inputs
        
        print("Band configuration:")
        print(f"   Total bands: {self.total_bands} (0-{self.total_bands-1})")
        print(f"   Input bands: {self.input_bands}")
        print(f"   Target models: {self.target_model_count}")
    
    def load_source_strategy_info(self):
        """Load summary info from the source strategy (trained bands and performance)."""
        print(f"Loading source strategy info: {self.source_strategy}")
        
        summary_file = os.path.join(self.source_dir, "training_summary.json")
        self.trained_bands = []
        self.band_performance = {}
        
        print(f"Looking for summary file: {summary_file}")
        print(f"   Exists: {os.path.exists(summary_file)}")
        
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    self.strategy_data = json.load(f)
                
                print(f"Summary keys: {list(self.strategy_data.keys())}")
                
                # Extract trained bands
                if 'training_results' in self.strategy_data:
                    for band_key, result in self.strategy_data['training_results'].items():
                        band_num = int(band_key.split('_')[1])
                        self.trained_bands.append(band_num)
                        self.band_performance[band_num] = {
                            'rmse': result.get('rmse', float('inf')),
                            'mae': result.get('mae', float('inf')),
                            'model_path': os.path.join(self.models_dir, f"band_{band_num:03d}_best_model.pth")
                        }
                
                print(f"Loaded {len(self.trained_bands)} bands from training summary")
                
            except Exception as e:
                print(f"Failed to read training summary: {e}")
                self.strategy_data = {}
        
        # Fallback: scan model files if no summary or no bands listed
        if not self.trained_bands:
            print("Scanning model files...")
            model_pattern = os.path.join(self.models_dir, "band_*_best_model.pth")
            print(f"Glob pattern: {model_pattern}")
            
            model_files = glob.glob(model_pattern)
            print(f"Found {len(model_files)} model files")
            
            if model_files:
                print("First 5 model files:")
                for model_file in model_files[:5]:
                    print(f"   - {os.path.basename(model_file)}")
            
            for model_file in model_files:
                try:
                    filename = os.path.basename(model_file)
                    band_num = int(filename.split('_')[1])
                    self.trained_bands.append(band_num)
                    self.band_performance[band_num] = {
                        'rmse': float('inf'),
                        'mae': float('inf'),
                        'model_path': model_file
                    }
                except Exception as e:
                    print(f"Failed to parse file name {filename}: {e}")
                    continue
            
            print(f"Loaded {len(self.trained_bands)} bands from file scan")
        
        self.trained_bands = sorted(self.trained_bands)
        
        print(f"Trained bands (first 10): {self.trained_bands[:10]}{'...' if len(self.trained_bands) > 10 else ''}")
        print(f"Band range: {min(self.trained_bands) if self.trained_bands else 'N/A'} - {max(self.trained_bands) if self.trained_bands else 'N/A'}")
    
    def find_best_source_model(self, target_band):
        """Find the best source model for a given target band."""
        # If the band is already trained (cascade-trained), reuse it directly
        if target_band in self.trained_bands:
            return self.band_performance[target_band]['model_path'], 1.0, target_band
        
        # Input bands are not to be reconstructed
        if target_band in self.input_bands:
            raise ValueError(f"Band {target_band} is an input band and does not require reconstruction")
        
        best_similarity = -1.0
        best_source_band = None
        best_model_path = None
        
        # Search among all trained bands
        for source_band in self.trained_bands:
            spectral_sim = get_spectral_similarity(target_band, source_band)
            performance_weight = 1.0 / (1.0 + self.band_performance[source_band]['rmse'])
            combined_score = 0.7 * spectral_sim + 0.3 * performance_weight
            
            if combined_score > best_similarity:
                best_similarity = combined_score
                best_source_band = source_band
                best_model_path = self.band_performance[source_band]['model_path']
        
        if best_model_path and os.path.exists(best_model_path):
            return best_model_path, best_similarity, best_source_band
        else:
            return None, 0.0, None
    
    def get_fine_tuning_params(self, similarity_score, target_band, source_band):
        """Derive fine-tuning hyperparameters based on band similarity and distance."""
        base_params = {
            'epochs': 50,
            'lr': 1e-4,
            'weight_decay': 5e-6,
            'warmup_epochs': 5,
            'gradient_clip': 1.0,
            'accumulation_steps': 1
        }
        
        # Adjust by similarity
        if similarity_score > 0.8:
            base_params.update({'epochs': 30, 'lr': 5e-5, 'warmup_epochs': 3})
        elif similarity_score > 0.6:
            base_params.update({'epochs': 50, 'lr': 1e-4, 'warmup_epochs': 5})
        else:
            base_params.update({'epochs': 80, 'lr': 2e-4, 'warmup_epochs': 8})
        
        # Adjust by spectral distance
        band_distance = abs(target_band - source_band)
        if band_distance < 5:
            base_params['epochs'] = max(20, int(base_params['epochs'] * 0.7))
        elif band_distance > 50:
            base_params['epochs'] = int(base_params['epochs'] * 1.3)
        
        return base_params
    
    def run_full_spectrum_fine_tuning(self, device='cuda', batch_size=8, start_band=0, end_band=223, dry_run=True):
        """Plan or execute full-spectrum fine-tuning from source strategy to 219 target bands."""
        print(f"Starting full-spectrum fine-tuning: {self.source_strategy} → {self.output_strategy}")
        
        all_bands = list(range(start_band, end_band + 1))  # 0..223 inclusive
        target_bands = [band for band in all_bands if band not in self.input_bands]
        missing_bands = [band for band in target_bands if band not in self.trained_bands]
        
        print("Band statistics:")
        print(f"   All bands count: {len(all_bands)} (range {start_band}-{end_band})")
        print(f"   Input bands: {len(self.input_bands)} {self.input_bands}")
        print(f"   Target bands to reconstruct: {len(target_bands)} (224 total - 5 inputs = 219)")
        print(f"   Already trained (cascade): {len(self.trained_bands)} (includes any input bands present: {[b for b in self.trained_bands if b in self.input_bands]})")
        print(f"   Bands to fine-tune: {len(missing_bands)}")
        
        expected_total_models = 219
        predicted_total = len(self.trained_bands) + len(missing_bands)
        
        print("Model count validation:")
        print(f"   Expected total models: {expected_total_models} (224 bands - 5 inputs)")
        print(f"   Predicted total (trained + to fine-tune): {predicted_total}")
        
        if predicted_total == expected_total_models:
            print("Count check passed")
        else:
            difference = abs(predicted_total - expected_total_models)
            print(f"Count mismatch, difference: {difference}")
            if predicted_total < expected_total_models:
                print(f"   Additional bands required: {difference}")
            else:
                print(f"   Exceeding expected by: {difference}")
        
        if not missing_bands:
            print("All 219 models are already available")
            return {}
        
        print(f"Bands requiring fine-tuning (first 20 shown): {missing_bands[:20]}{'...' if len(missing_bands) > 20 else ''}")
        
        # Plan for each missing band
        fine_tuning_plan = {}
        total_bands = len(missing_bands)
        
        for i, target_band in enumerate(missing_bands, 1):
            print(f"\nPlanning band {i}/{total_bands}: Band {target_band}")
            try:
                source_model_path, similarity, source_band = self.find_best_source_model(target_band)
                
                if source_model_path:
                    params = self.get_fine_tuning_params(similarity, target_band, source_band)
                    fine_tuning_plan[target_band] = {
                        'source_band': source_band,
                        'source_model_path': source_model_path,
                        'similarity': similarity,
                        'params': params
                    }
                    print(f"   Plan: Band {target_band:3d} ← Band {source_band:3d} "
                          f"(similarity: {similarity:.3f}, epochs: {params['epochs']}, lr: {params['lr']:.6f})")
                else:
                    print(f"   Could not find a suitable source model for band {target_band:3d}")
                    
            except Exception as e:
                print(f"   Planning failed for band {target_band:3d}: {e}")
        
        print("\nFine-tuning plan completed:")
        print(f"   Executable plans: {len(fine_tuning_plan)}")
        print(f"   Unplanned bands: {total_bands - len(fine_tuning_plan)}")
        
        if dry_run:
            print("\nDry run complete")
            print("Use --run_actual to execute fine-tuning")
            
            # Save plan
            plan_file = os.path.join(self.output_dir, "fine_tuning_plan.json")
            with open(plan_file, 'w') as f:
                json.dump({
                    'source_strategy': self.source_strategy,
                    'total_missing_bands': total_bands,
                    'executable_plans': len(fine_tuning_plan),
                    'plan_details': fine_tuning_plan
                }, f, indent=2, default=str)
            
            print(f"Plan saved to: {plan_file}")
            return fine_tuning_plan
        
        # Execute
        print("\nStarting actual fine-tuning...")
        return self.execute_fine_tuning(fine_tuning_plan, device, batch_size)
    
    def execute_fine_tuning(self, fine_tuning_plan, device, batch_size):
        """Execute the fine-tuning process for all bands in the plan."""
        print(f"Executing fine-tuning for {len(fine_tuning_plan)} bands")
        
        results = {}
        failed_bands = []
        start_time = time.time()
        
        try:
            writer = SummaryWriter(self.output_logs_dir)
            
            for i, (target_band, plan) in enumerate(fine_tuning_plan.items(), 1):
                print(f"\n{'='*60}")
                print(f"Fine-tuning {i}/{len(fine_tuning_plan)}: Band {target_band}")
                print(f"{'='*60}")
                
                try:
                    result = self.fine_tune_single_band(
                        target_band, plan, device, batch_size, writer
                    )
                    
                    if result:
                        results[f"band_{target_band:03d}"] = result
                        print(f"Band {target_band} finished successfully")
                    else:
                        failed_bands.append(target_band)
                        print(f"Band {target_band} failed")
                
                except Exception as e:
                    print(f"Band {target_band} encountered an error: {e}")
                    failed_bands.append(target_band)
                    continue
            
            writer.close()
            
        except Exception as e:
            print(f"Fine-tuning execution failed: {e}")
        
        total_time = time.time() - start_time
        
        # Save results
        self.save_fine_tuning_results(results, failed_bands, total_time)
        
        print("\nFine-tuning finished")
        print(f"Success: {len(results)} bands")
        print(f"Failed: {len(failed_bands)} bands")
        print(f"Total time: {total_time/3600:.2f} hours")
        
        return results
    
    def fine_tune_single_band(self, target_band, plan, device, batch_size, writer):
        print(f"Start fine-tuning for band {target_band}")
        print(f"   Source band: {plan['source_band']}")
        print(f"   Similarity: {plan['similarity']:.4f}")
        
        start_time = time.time()
        
        try:
            # 1) Load source model
            print(f"Loading source model: {plan['source_model_path']}")
            checkpoint = torch.load(plan['source_model_path'], map_location=device)
            
            # 2) Create model consistent with training setup
            model = SpecSwin_SingleBand(
                in_channels=16,
                img_size=(64, 64)
            ).to(device)
            print("Model created")
            
            # 3) Load pretrained weights
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Source model weights loaded")
            
            # 4) Create data loaders
            print("Creating data loaders...")
            train_loader, val_loader, test_loader = create_single_band_loaders(
                target_band_idx=target_band,
                batch_size=batch_size,
                num_workers=2
            )
            print(f"Train size: {len(train_loader.dataset)}")
            print(f"Val size: {len(val_loader.dataset)}")
            
            # 5) Normalization parameters
            dataset = train_loader.dataset
            if hasattr(dataset, 'dataset'):
                norm_params = dataset.dataset.get_norm_params()
            else:
                norm_params = dataset.get_norm_params()
            print(f"Normalization params available: {norm_params is not None}")
            
            # 6) Optimizer, loss, scheduler
            params = plan['params']
            optimizer = optim.AdamW(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=params['epochs']
            )
            
            # 7) Training loop
            print(f"Training for {params['epochs']} epochs...")
            best_val_loss = float('inf')
            best_model_path = os.path.join(
                self.output_models_dir, 
                f"band_{target_band:03d}_finetuned.pth"
            )
            
            for epoch in range(params['epochs']):
                # Train
                model.train()
                train_loss = 0.0
                train_batches = 0
                
                for batch_idx, (inputs, targets) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['epochs']} - Train")
                ):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip'])
                    optimizer.step()
                    train_loss += loss.item()
                    train_batches += 1
                
                # Validate
                model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for inputs, targets in tqdm(val_loader, desc="Validation"):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_train_loss = train_loss / max(train_batches, 1)
                avg_val_loss = val_loss / max(val_batches, 1)
                scheduler.step()
                
                # Log to TensorBoard
                if writer:
                    writer.add_scalar(f'Loss/Train_Band_{target_band}', avg_train_loss, epoch)
                    writer.add_scalar(f'Loss/Val_Band_{target_band}', avg_val_loss, epoch)
                    writer.add_scalar(f'LR/Band_{target_band}', optimizer.param_groups[0]['lr'], epoch)
                
                print(f"Epoch {epoch+1:3d}: Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.8f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'target_band': target_band,
                        'source_band': plan['source_band'],
                        'similarity': plan['similarity'],
                        'source_strategy': self.source_strategy,
                        'fine_tuning_time': time.time() - start_time
                    }, best_model_path)
                    print(f"Saved best model: {best_model_path}")
            
            # 8) Final evaluation
            print("Final evaluation on test set...")
            final_metrics = evaluate_model(model, test_loader, device, norm_params)
            training_time = time.time() - start_time
            
            result = {
                'target_band': target_band,
                'source_band': plan['source_band'],
                'similarity': plan['similarity'],
                'rmse': final_metrics.get('rmse', best_val_loss**0.5),
                'mae': final_metrics.get('mae', best_val_loss),
                'best_val_loss': best_val_loss,
                'training_time': training_time,
                'epochs_completed': params['epochs'],
                'model_path': best_model_path
            }
            
            # Include original-scale metrics if available
            if 'rmse_original' in final_metrics:
                result.update({
                    'rmse_original': final_metrics['rmse_original'],
                    'mae_original': final_metrics['mae_original'],
                    'norm_range': final_metrics.get('norm_range')
                })
            
            print(f"Band {target_band} training complete")
            print(f"   Best validation loss: {best_val_loss:.6f}")
            print(f"   Normalized RMSE: {final_metrics.get('rmse', float('nan')):.6f}")
            print(f"   Normalized MAE: {final_metrics.get('mae', float('nan')):.6f}")
            if 'rmse_original' in final_metrics:
                print(f"   Original-scale RMSE: {final_metrics['rmse_original']:.2f}")
                print(f"   Original-scale MAE: {final_metrics['mae_original']:.2f}")
            print(f"   Training time (minutes): {training_time/60:.1f}")
            
            return result
            
        except Exception as e:
            print(f"Band {target_band} training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_fine_tuning_results(self, results, failed_bands, total_time):
        """Save fine-tuning summary results to disk."""
        final_results = {
            'source_strategy': self.source_strategy,
            'output_strategy': self.output_strategy,
            'timestamp': datetime.now().isoformat(),
            'total_time_hours': total_time / 3600,
            'successful_bands': len(results),
            'failed_bands': len(failed_bands),
            'results': results,
            'failed_band_list': failed_bands
        }
        
        results_file = os.path.join(self.output_dir, "fine_tuning_results.json")
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
