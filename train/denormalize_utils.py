import torch
import numpy as np

def denormalize_prediction(normalized_pred, norm_params):
    """
    Denormalize prediction results
    Args:
        normalized_pred: Prediction values in the range [0, 1]
        norm_params: Dictionary containing p1 and p99
    Returns:
        Prediction values in the original range
    """
    p1, p99 = norm_params['p1'], norm_params['p99']
    original = normalized_pred * (p99 - p1) + p1
    return original

def denormalize_batch(normalized_batch, norm_params_list):
    """
    Args:
        normalized_batch: [B, C, H, W] normalized tensor
        norm_params_list: List of normalization parameters for each band
    Returns:
        Denormalized tensor
    """
    if isinstance(normalized_batch, torch.Tensor):
        normalized_batch = normalized_batch.cpu().numpy()
    
    denormalized = np.zeros_like(normalized_batch)
    
    for b in range(normalized_batch.shape[0]):
        for c in range(normalized_batch.shape[1]):
            if c < len(norm_params_list):
                denormalized[b, c] = denormalize_prediction(
                    normalized_batch[b, c], 
                    norm_params_list[c]
                )
    
    return torch.tensor(denormalized)
