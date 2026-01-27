
# backbone: dinov2, dinov3, sam
# dataset: spair71k, pf-pascal, pf-willow
# split: train, val, test

import torch
from .common_inference import  CONFIGURATION_INF

def argmax_strategy(similarity_map, best_idx, Hf, Wf):
    """
    Simple argmax prediction strategy.
    
    Args:
        similarity_map: Computed similarity map
        best_idx: Index of maximum similarity
        Hf, Wf: Height and width of feature map
        
    Returns:
        torch.Tensor: Predicted coordinates [x, y]
    """
    return torch.tensor([best_idx % Wf, best_idx // Wf], device=CONFIGURATION_INF["DEVICE"])
