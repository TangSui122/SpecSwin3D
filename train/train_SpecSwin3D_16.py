import sys
import os
import json
import contextlib
from datetime import datetime
from pathlib import Path
import glob
import argparse

def resolve_paths():
    """
    Resolve project paths using environment variables with safe defaults:
      PROJECT_ROOT   -> default: parent directory of this file
      DATA_DIR       -> default: <PROJECT_ROOT>/dataset
      CHECKPOINT_DIR -> default: <PROJECT_ROOT>/checkpoints
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.environ.get("PROJECT_ROOT", os.path.dirname(this_dir))
    data_dir = os.environ.get("DATA_DIR", os.path.join(project_root, "dataset"))
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", os.path.join(project_root, "checkpoints"))
    return project_root, data_dir, checkpoint_dir

PROJECT_ROOT, BASE_DIR, CHECKPOINT_DIR = resolve_paths()

# Allow relative imports from project root if needed
sys.path.append(PROJECT_ROOT)

# -------------------------------
# Imports (ML stack)
# -------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# external (you should have these in your env)
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from monai.networks.nets import SwinUNETR

# If you have these utilities in your project, they will be imported from PROJECT_ROOT
# Otherwise, replace/implement as needed.
try:
    from denormalize_utils import denormalize_prediction
except Exception:
    # Fallback if utility not available
    def denormalize_prediction(arr, norm_params):
        # Identity fallback; replace with your real de-normalization
        return arr

# -------------------------------
# Model
# -------------------------------
class SpecSwin_SingleBand(nn.Module):
    """Single-band spectral reconstruction model based on SwinUNETR."""
    def __init__(self, in_channels=16, img_size=(128, 128), spatial_dims=2):
        super().__init__()
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=1,
            feature_size=48,
            spatial_dims=spatial_dims,
            use_checkpoint=False
        )
    
    def forward(self, x):
        return self.model(x)

# -------------------------------
# Dataset
# -------------------------------
class SingleBandDataset(Dataset):
    """Single-band spectral reconstruction dataset (2D or pseudo-3D)."""
    def __init__(self, input_dir, label_dir, target_band_idx, use_3d=False):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.target_band_idx = target_band_idx
        self.use_3d = use_3d
        
        # Input bands that are not reconstructed
        self.input_bands = [30, 20, 9, 40, 52]
        if target_band_idx in self.input_bands:
            raise ValueError(
                f"Target band {target_band_idx} is an input band and cannot be a reconstruction target. "
                f"Input bands: {self.input_bands}"
            )
        
        self.input_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "*.pt")))
        assert len(self.input_files) == len(self.label_files), \
            f"Input/label count mismatch: {len(self.input_files)} vs {len(self.label_files)}"
        
        # Try to read normalization parameters from a sample label
        self.norm_params = None
        sample_label = torch.load(self.label_files[0], map_location='cpu')
        if isinstance(sample_label, dict) and 'norm_params' in sample_label:
            label_band_indices = sample_label.get('band_indices', [])
            if target_band_idx in label_band_indices:
                pos = label_band_indices.index(target_band_idx)
                if pos < len(sample_label['norm_params']):
                    self.norm_params = sample_label['norm_params'][pos]
        
        print(f"Single-band dataset: {len(self.input_files)} samples, target band: {target_band_idx}")
    
    def get_norm_params(self):
        return self.norm_params
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_data = torch.load(self.input_files[idx], map_location='cpu')
        label_data = torch.load(self.label_files[idx], map_location='cpu')
        
        input_tensor = input_data['input'] if isinstance(input_data, dict) else input_data
        label_tensor = label_data['label'] if isinstance(label_data, dict) else label_data
        
        input_tensor = input_tensor.float()
        label_tensor = label_tensor.float()
        
        # 2D default: input (16, 128, 128)
        # If use_3d: convert to (1, 128, 128, 32) by repeating depth
        if self.use_3d:
            if input_tensor.dim() == 3 and input_tensor.shape[0] == 16:
                input_tensor = input_tensor.permute(1, 2, 0)  # (128, 128, 16)
                input_tensor = torch.cat([input_tensor, input_tensor], dim=2)  # (128, 128, 32)
                input_tensor = input_tensor.unsqueeze(0)  # (1, 128, 128, 32)
        
        # Select target band from label
        if isinstance(label_data, dict) and 'band_indices' in label_data:
            label_band_indices = label_data['band_indices']
            if self.target_band_idx in label_band_indices:
                band_pos = label_band_indices.index(self.target_band_idx)
                if band_pos < label_tensor.shape[0]:
                    target_band = label_tensor[band_pos:band_pos+1]
                else:
                    raise ValueError(f"Target band position {band_pos} exceeds label tensor size {label_tensor.shape[0]}")
            else:
                raise ValueError(
                    f"Target band {self.target_band_idx} is not present in label data. "
                    f"It might be one of the input bands {self.input_bands}"
                )
        else:
            if self.target_band_idx < label_tensor.shape[0]:
                target_band = label_tensor[self.target_band_idx:self.target_band_idx+1]
            else:
                raise ValueError(f"Target band {self.target_band_idx} exceeds label tensor size {label_tensor.shape[0]}")
        
        if self.use_3d:
            # Convert target to (1, 128, 128, 32)
            if target_band.dim() == 2:
                target_band = target_band.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 32)
            elif target_band.dim() == 3:
                target_band = target_band.unsqueeze(-1).repeat(1, 1, 1, 32)
        
        return input_tensor, target_band

# -------------------------------
# Data Loaders
# -------------------------------
def create_single_band_loaders(
    target_band_idx,
    batch_size=16,
    num_workers=8,
    data_type='with_indices',
    use_3d=False
):
    """
    Create train/val/test loaders using soft-coded directories.
    DATA_DIR layout (defaults): 
      <DATA_DIR>/
        input_restacked_16/
        input_restacked_16_with_indices/
        input_restacked_16_repeated/
        label/
    """
    # Select subdir by data type
    if data_type == 'with_indices':
        input_dir_name = "input_restacked_16_with_indices"
        data_desc = "Vegetation-index enhanced data"
    elif data_type == 'repeated':
        input_dir_name = "input_restacked_16_repeated"
        data_desc = "Repeated pattern 16-channel data [0,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]"
    else:
        input_dir_name = "input_restacked_16"
        data_desc = "Original 16-channel data"
    
    input_dir = os.path.join(BASE_DIR, input_dir_name)
    label_dir = os.path.join(BASE_DIR, "label")
    
    print(f"DATA_DIR: {BASE_DIR}")
    print(f"Using {data_desc}: {input_dir}")
    print(f"Using labels: {label_dir}")
    
    if not (os.path.exists(input_dir) and os.path.exists(label_dir)):
        raise FileNotFoundError(
            f"Required data directories not found.\n"
            f"  INPUT: {input_dir}\n"
            f"  LABEL: {label_dir}\n"
            f"Ensure DATA_DIR is correct (env var) or the default structure exists."
        )
    
    # Basic file check
    n_inputs = len(glob.glob(os.path.join(input_dir, "*.pt")))
    n_labels = len(glob.glob(os.path.join(label_dir, "*.pt")))
    print(f"Found files -> input: {n_inputs}, label: {n_labels}")
    if n_inputs == 0 or n_labels == 0:
        raise FileNotFoundError("No .pt files found in input or label directory.")
    
    full_dataset = SingleBandDataset(input_dir, label_dir, target_band_idx, use_3d=use_3d)
    n = len(full_dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Worker settings
    num_workers = 0 if os.name == 'nt' else num_workers
    pin_memory = False
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader

# -------------------------------
# Similarity & Pretrained Selection
# -------------------------------
def get_spectral_similarity(b1, b2):
    """Heuristic spectral similarity between two bands."""
    groups = {'visible': (0, 79), 'nir': (80, 159), 'swir': (160, 218)}
    def group_of(b):
        for g, (s, e) in groups.items():
            if s <= b <= e:
                return g
        return 'unknown'
    g1, g2 = group_of(b1), group_of(b2)
    dist = abs(b1 - b2)
    if g1 == g2:
        return 1.0 / (1.0 + dist * 0.1)
    return 1.0 / (1.0 + dist * 0.2)

def find_best_pretrained_model(
    target_band_idx, trained_bands, checkpoint_dir,
    current_strategy, use_pretrained=False, pretrained_strategy=None
):
    """Find a pretrained model path based on band similarity (soft-coded paths)."""
    if not use_pretrained or not trained_bands:
        print(f"Band {target_band_idx}: train from scratch (no pretrained).")
        return None
    
    # Rank by similarity
    sims = sorted(
        [(b, get_spectral_similarity(target_band_idx, b)) for b in trained_bands],
        key=lambda x: x[1], reverse=True
    )
    best_band = sims[0][0]
    
    # Determine search strategies
    if pretrained_strategy:
        strategies = [pretrained_strategy]
        print(f"Searching pretrained only in strategy: {pretrained_strategy}")
    else:
        try:
            strategies = [d for d in os.listdir(checkpoint_dir)
                          if os.path.isdir(os.path.join(checkpoint_dir, d))]
        except Exception:
            strategies = [current_strategy]
        # Current strategy first
        if current_strategy in strategies:
            strategies.remove(current_strategy)
            strategies = [current_strategy] + strategies
        print(f"Pretrained search order: {strategies}")
    
    for strat in strategies:
        model_path = os.path.join(checkpoint_dir, strat, "models", f"band_{best_band:03d}_best_model.pth")
        if os.path.exists(model_path):
            print(f"Band {target_band_idx} will use pretrained from band {best_band} (strategy: {strat})")
            print(f"Similarity: {sims[0][1]:.3f}")
            print(f"Model path: {model_path}")
            return model_path
    
    print(f"Band {target_band_idx}: no suitable pretrained model found; train from scratch.")
    return None

# -------------------------------
# Training Params (Cascade)
# -------------------------------
def get_cascade_training_params(target_band_idx, pretrained_available, cascade_level):
    base_epochs = 80
    base_lr = 2e-4
    if not pretrained_available:
        return {
            'epochs': base_epochs,
            'lr': base_lr,
            'weight_decay': 5e-6,
            'warmup_epochs': 10,
            'gradient_clip': 1.0,
            'accumulation_steps': 2
        }
    lr_decay = 0.7 ** cascade_level
    epoch_decay = max(0.5, 0.9 ** cascade_level)
    return {
        'epochs': max(40, int(base_epochs * epoch_decay)),
        'lr': base_lr * lr_decay,
        'weight_decay': 5e-6,
        'warmup_epochs': 5,
        'gradient_clip': 1.0,
        'accumulation_steps': 1
    }

# -------------------------------
# Train One Band
# -------------------------------
def train_single_band_model(
    target_band_idx, cascade_level, batch_size, device,
    pretrained_model_path, strategy_dir, writer,
    data_type='with_indices', use_3d=False
):
    print(f"\nTraining Band {target_band_idx} (Level {cascade_level}) - {'3D' if use_3d else '2D'}")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    train_loader, val_loader, test_loader = create_single_band_loaders(
        target_band_idx, batch_size=batch_size,
        data_type=data_type, use_3d=use_3d
    )
    
    params = get_cascade_training_params(target_band_idx, pretrained_model_path is not None, cascade_level)
    
    # Build model
    if use_3d:
        model = SpecSwin_SingleBand(in_channels=1, img_size=(128, 128, 32), spatial_dims=3).to(device)
    else:
        model = SpecSwin_SingleBand(in_channels=16, img_size=(128, 128), spatial_dims=2).to(device)
    
    # Load pretrained
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained weights: {pretrained_model_path}")
        try:
            checkpoint = torch.load(pretrained_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Pretrained weights loaded.")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
    
    # Setup optimization
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scaler = None  # set to torch.cuda.amp.GradScaler() if you want mixed precision
    
    warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=params['warmup_epochs'])
    main_sched = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=params['lr'] * 2,
        total_steps=params['epochs'] - params['warmup_epochs'],
        pct_start=0.3, anneal_strategy='cos'
    )
    
    best_val = float('inf')
    best_model_path = os.path.join(strategy_dir, f"band_{target_band_idx:03d}_best_model.pth")
    
    print(f"Training params: epochs={params['epochs']}, lr={params['lr']:.6f}")
    
    for epoch in range(1, params['epochs'] + 1):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Band {target_band_idx} L{cascade_level} Epoch {epoch}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if epoch == 1 and batch_idx == 0:
                print(f"[DEBUG] Train shapes: inputs={tuple(inputs.shape)}, targets={tuple(targets.shape)}")
            
            with torch.cuda.amp.autocast() if scaler else contextlib.nullcontext():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if 'accumulation_steps' in params:
                    loss = loss / params['accumulation_steps']
            
            if scaler:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % params.get('accumulation_steps', 1) == 0:
                    if 'gradient_clip' in params:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % params.get('accumulation_steps', 1) == 0:
                    if 'gradient_clip' in params:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip'])
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * params.get('accumulation_steps', 1)
            pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'LR': f'{optimizer.param_groups[0]["lr"]:.7f}'})
        
        avg_train = train_loss / max(1, len(train_loader))
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        avg_val = val_loss / max(1, len(val_loader))
        
        # LR schedule
        if epoch <= params['warmup_epochs']:
            warmup_sched.step()
        else:
            main_sched.step()
        
        # TensorBoard logs
        global_step = cascade_level * 1000 + epoch
        if writer:
            writer.add_scalar(f'Loss/Train_Band_{target_band_idx}', avg_train, global_step)
            writer.add_scalar(f'Loss/Val_Band_{target_band_idx}', avg_val, global_step)
            writer.add_scalar(f'LR/Band_{target_band_idx}', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar(f'CascadeLevel/Band_{target_band_idx}', cascade_level, global_step)
        
        print(f"Epoch {epoch}/{params['epochs']}: Train={avg_train:.6f}, Val={avg_val:.6f}")
        
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val,
                'target_band_idx': target_band_idx,
                'cascade_level': cascade_level,
                'train_params': params
            }, best_model_path)
    
    # Final test
    print(f"Band {target_band_idx} final testing...")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Norm params
    dataset = train_loader.dataset
    if hasattr(dataset, 'dataset'):
        norm_params = dataset.dataset.get_norm_params()
    else:
        norm_params = dataset.get_norm_params()
    
    results = evaluate_model(model, test_loader, device, norm_params)
    print(f"Band {target_band_idx} test results:")
    print(f"   RMSE: {results['rmse']:.6f}")
    print(f"   MAE: {results['mae']:.6f}")
    if 'rmse_original' in results:
        print(f"   Original-scale RMSE: {results['rmse_original']:.2f}")
        print(f"   Original-scale MAE: {results['mae_original']:.2f}")
    
    return best_model_path, results

# -------------------------------
# Evaluation
# -------------------------------
def evaluate_model(model, test_loader, device, norm_params):
    criterion = nn.MSELoss()
    test_loss = 0.0
    mse_sum = 0.0
    mae_sum = 0.0
    rmse_orig_sum = 0.0
    mae_orig_sum = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            test_loss += criterion(outputs, targets).item()
            mse_sum += nn.MSELoss()(outputs, targets).item() * inputs.size(0)
            mae_sum += nn.L1Loss()(outputs, targets).item() * inputs.size(0)
            
            if norm_params is not None:
                for b in range(outputs.shape[0]):
                    pred_denorm = denormalize_prediction(outputs[b, 0].cpu().numpy(), norm_params)
                    target_denorm = denormalize_prediction(targets[b, 0].cpu().numpy(), norm_params)
                    rmse_o = np.sqrt(np.mean((pred_denorm - target_denorm) ** 2))
                    mae_o = np.mean(np.abs(pred_denorm - target_denorm))
                    rmse_orig_sum += rmse_o
                    mae_orig_sum += mae_o
            
            n_samples += inputs.size(0)
    
    results = {
        'test_loss': test_loss / max(1, len(test_loader)),
        'mse': mse_sum / max(1, n_samples),
        'mae': mae_sum / max(1, n_samples),
        'rmse': np.sqrt(mse_sum / max(1, n_samples))
    }
    if norm_params is not None and n_samples > 0:
        results.update({
            'rmse_original': rmse_orig_sum / n_samples,
            'mae_original': mae_orig_sum / n_samples,
            'norm_range': [norm_params.get('p1'), norm_params.get('p99')] if isinstance(norm_params, dict) else None
        })
    return results

# -------------------------------
# Strategy IO (Soft-coded)
# -------------------------------
def load_custom_strategy_from_txt(txt_file):
    """Load custom cascade strategy mapping level->list[bands] from a .txt file."""
    strategy_data = {}
    strategy_name = "custom"
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.startswith('strategy_name:'):
                strategy_name = s.split(':', 1)[1].strip()
                continue
            if s.startswith('level_'):
                parts = s.split(':', 1)
                level = int(parts[0].split('_')[1])
                bands_str = parts[1].strip()
                if bands_str.startswith('[') and bands_str.endswith(']'):
                    bands_str = bands_str[1:-1]
                bands = [int(x.strip()) for x in bands_str.split(',') if x.strip()]
                strategy_data[level] = bands
    return strategy_data, strategy_name

def design_cascade_strategy(strategy_type="physical", custom_txt_file=None):
    """
    Build cascade strategy:
      - 'physical' and 'uniform' are built-in.
      - 'variance_importance' / 'correlation_importance' / 'mutual_info_importance' / 'spectral_physics_importance'
        are loaded from '<CHECKPOINT_DIR>/importance_analysis/strategies/*.txt'
      - 'custom' loads from a user-specified txt file.
    """
    txt_based = ["variance_importance", "correlation_importance", "mutual_info_importance", "spectral_physics_importance"]
    
    if strategy_type == "physical":
        return {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
            1: [15, 25, 27],
            2: [48, 50, 54, 67, 81, 83, 90],
            3: [108, 125, 135],
            4: [155, 162, 175, 185, 189, 210, 218]
        }
    if strategy_type == "uniform":
        selected = [int(i * 218 / 28) for i in range(29)]
        return {
            0: selected[0:9],
            1: selected[9:12],
            2: selected[12:19],
            3: selected[19:22],
            4: selected[22:29]
        }
    if strategy_type == "custom":
        if not custom_txt_file:
            raise ValueError("When strategy=custom, you must provide --custom_txt.")
        strategy_data, _ = load_custom_strategy_from_txt(custom_txt_file)
        return strategy_data
    
    if strategy_type in txt_based:
        strategies_dir = os.path.join(CHECKPOINT_DIR, "importance_analysis", "strategies")
        file_map = {
            "variance_importance": "variance_strategy.txt",
            "correlation_importance": "correlation_strategy.txt",
            "mutual_info_importance": "mutual_info_strategy.txt",
            "spectral_physics_importance": "spectral_physics_strategy.txt"
        }
        strategy_file = os.path.join(strategies_dir, file_map[strategy_type])
        if not os.path.exists(strategy_file):
            raise FileNotFoundError(
                f"Strategy file not found: {strategy_file}\n"
                f"Set CHECKPOINT_DIR env var or create the default directory structure."
            )
        strategy_data, _ = load_custom_strategy_from_txt(strategy_file)
        return strategy_data
    
    raise ValueError(f"Unknown strategy: {strategy_type}. "
                     f"Available: ['physical','uniform',{txt_based},'custom']")

# -------------------------------
# Utility Saves
# -------------------------------
def save_cascade_record(cascade_levels, strategy_name, base_dir):
    """Append cascade mapping to '<base_dir>/cascade_levels.txt'."""
    record_file = os.path.join(base_dir, "cascade_levels.txt")
    mode = 'a' if os.path.exists(record_file) else 'w'
    with open(record_file, mode, encoding='utf-8') as f:
        if mode == 'w':
            f.write("# Cascade training band-level records\n")
            f.write(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Format: Band_xxx -> Level_x (Strategy: strategy_name)\n")
            f.write("=" * 70 + "\n\n")
        f.write(f"# Strategy: {strategy_name.upper()}\n")
        f.write(f"# Total levels: {len(cascade_levels)}\n")
        f.write(f"# Total bands: {sum(len(b) for b in cascade_levels.values())}\n")
        f.write("-" * 50 + "\n")
        for level in sorted(cascade_levels.keys()):
            bands = cascade_levels[level]
            for band in sorted(bands):
                f.write(f"Band_{band:03d} -> Level_{level} (Strategy: {strategy_name})\n")
        f.write("\n" + "=" * 70 + "\n\n")
    print(f"Cascade record appended to: {record_file}")
    return record_file

def save_training_summary(strategy_name, base_dir, cascade_levels, training_results):
    """Save training summary JSON to '<base_dir>/<strategy_name>/training_summary.json'."""
    strategy_dir = os.path.join(base_dir, strategy_name)
    os.makedirs(strategy_dir, exist_ok=True)
    summary_file = os.path.join(strategy_dir, "training_summary.json")
    summary = {
        'strategy': strategy_name,
        'timestamp': datetime.now().isoformat(),
        'cascade_levels': cascade_levels,
        'total_bands': sum(len(b) for b in cascade_levels.values()),
        'total_levels': len(cascade_levels),
        'training_results': training_results,
        'best_bands': {
            'lowest_rmse': min(training_results.items(), key=lambda x: x[1].get('rmse', float('inf')))[0] if training_results else None,
            'lowest_mae': min(training_results.items(), key=lambda x: x[1].get('mae', float('inf')))[0] if training_results else None
        }
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Training summary saved to: {summary_file}")
    return summary_file

# -------------------------------
# Main
# -------------------------------
def main():
    print("Resolved paths (soft-coded):")
    print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"  DATA_DIR     : {BASE_DIR}")
    print(f"  CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    
    parser = argparse.ArgumentParser(description='Cascade spectral reconstruction training (soft-coded paths)')
    parser.add_argument('--strategy', type=str, default='physical',
                        choices=['physical', 'uniform', 'variance_importance',
                                 'correlation_importance', 'mutual_info_importance',
                                 'spectral_physics_importance', 'custom'],
                        help='Cascade strategy type')
    parser.add_argument('--custom_txt', type=str, default=None,
                        help='Custom strategy txt file (required when --strategy custom)')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--start_level', type=int, default=0, help='Start cascade level')
    parser.add_argument('--end_level', type=int, default=4, help='End cascade level')
    parser.add_argument('--data_type', type=str, default='with_indices',
                        choices=['with_indices', 'original', 'repeated'],
                        help='Data variant to use')
    parser.add_argument('--use_pretrained', action='store_true', default=False,
                        help='Reuse pretrained band models if available')
    parser.add_argument('--pretrained_strategy', type=str, default=None,
                        help='Specific strategy to search for pretrained models')
    parser.add_argument('--force_retrain', action='store_true', default=False,
                        help='Retrain even if a model exists')
    parser.add_argument('--use_3d', action='store_true', default=False,
                        help='Enable pseudo-3D mode (1,128,128,32)')
    args = parser.parse_args()
    
    print("\nParsed parameters:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    # Load cascade strategy
    if args.strategy == "custom":
        if not args.custom_txt:
            raise ValueError("When strategy=custom, --custom_txt must be provided.")
        cascade_levels = design_cascade_strategy(args.strategy, args.custom_txt)
        _, strategy_name = load_custom_strategy_from_txt(args.custom_txt)
    else:
        cascade_levels = design_cascade_strategy(args.strategy)
        strategy_name = args.strategy
    
    # Strategy directory name with data-type suffix
    if args.data_type == 'with_indices':
        strategy_name_with_suffix = f"{strategy_name}_with_indices"
        data_desc = "Vegetation-index enhanced data"
    elif args.data_type == 'repeated':
        strategy_name_with_suffix = f"{strategy_name}_repeated"
        data_desc = "Repeated pattern 16-channel data"
    else:
        strategy_name_with_suffix = f"{strategy_name}_original"
        data_desc = "Original 16-channel data"
    
    strategy_dir = os.path.join(CHECKPOINT_DIR, strategy_name_with_suffix)
    models_dir = os.path.join(strategy_dir, "models")
    tensorboard_dir = os.path.join(strategy_dir, "tensorboard_logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    print(f"\nStrategy directory created:")
    print(f"  {strategy_dir}")
    print(f"  models/ -> {models_dir}")
    print(f"  tensorboard_logs/ -> {tensorboard_dir}")
    
    # TensorBoard writer
    try:
        writer = SummaryWriter(tensorboard_dir)
    except Exception as e:
        print(f"TensorBoard writer initialization failed: {e}")
        writer = None
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    print(f"\nStrategy details for {strategy_name}:")
    total = 0
    for lv in sorted(cascade_levels.keys()):
        bands = cascade_levels[lv]
        total += len(bands)
        print(f"  Level {lv}: {len(bands)} bands -> {bands}")
    print(f"  Total bands across levels: {total}")
    
    # Optional: record the cascade mapping
    try:
        save_cascade_record(cascade_levels, strategy_name, strategy_dir)
    except Exception as e:
        print(f"Warning: could not save cascade record: {e}")
    
    # Collect pretrained availability (soft-coded scanning)
    def scan_trained_bands(checkpoint_dir, strategy=None):
        bands = set()
        strategies = [strategy] if strategy else [
            d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        for s in strategies:
            model_dir = os.path.join(checkpoint_dir, s, "models")
            if not os.path.exists(model_dir):
                continue
            for f in os.listdir(model_dir):
                if f.startswith("band_") and f.endswith("_best_model.pth"):
                    try:
                        bands.add(int(f.split('_')[1]))
                    except Exception:
                        pass
        return sorted(bands)
    
    available_pretrained_bands = scan_trained_bands(CHECKPOINT_DIR) if args.use_pretrained else []
    print(f"\nPretrained bands available: {len(available_pretrained_bands)}")
    
    # Bands already trained for current strategy
    already_trained = scan_trained_bands(CHECKPOINT_DIR, strategy_name_with_suffix)
    print(f"Already-trained bands in current strategy: {len(already_trained)}")
    
    training_results = {}
    
    # Training loop over levels/bands
    for level in range(args.start_level, args.end_level + 1):
        if level not in cascade_levels:
            print(f"Level {level} not in cascade strategy; skipping.")
            continue
        bands = cascade_levels[level]
        print(f"\nStart training cascade level {level} with {len(bands)} bands")
        print(f"Bands: {bands}")
        
        for i, band_idx in enumerate(bands, start=1):
            try:
                if not args.force_retrain and band_idx in already_trained:
                    print(f"Skip band {band_idx} (already trained). Use --force_retrain to retrain.")
                    continue
                
                pretrained_path = find_best_pretrained_model(
                    band_idx,
                    available_pretrained_bands,
                    CHECKPOINT_DIR,
                    strategy_name_with_suffix,
                    use_pretrained=args.use_pretrained,
                    pretrained_strategy=args.pretrained_strategy
                )
                
                model_path, result = train_single_band_model(
                    target_band_idx=band_idx,
                    cascade_level=level,
                    batch_size=args.batch_size,
                    device=device,
                    pretrained_model_path=pretrained_path,
                    strategy_dir=models_dir,
                    writer=writer,
                    data_type=args.data_type,
                    use_3d=args.use_3d
                )
                
                training_results[f"band_{band_idx:03d}"] = result
                already_trained.append(band_idx)
                print(f"Band {band_idx} training complete.")
            
            except Exception as e:
                print(f"Band {band_idx} training failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Cascade level {level} finished.")
    
    # Save summary
    try:
        save_training_summary(strategy_name_with_suffix, CHECKPOINT_DIR, cascade_levels, training_results)
    except Exception as e:
        print(f"Warning: could not save training summary: {e}")
    
    print("\nAll cascade training completed.")
    print(f"Models directory: {models_dir}")
    print(f"Cascade record: {os.path.join(strategy_dir, 'cascade_levels.txt')}")
    print(f"Training summary: {os.path.join(strategy_dir, 'training_summary.json')}")
    print(f"TensorBoard logs: {tensorboard_dir}")

if __name__ == "__main__":
    main()
