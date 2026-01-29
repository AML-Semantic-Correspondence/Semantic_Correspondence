
# backbone: dinov2, dinov3, sam
# dataset: spair71k, pf-pascal, pf-willow
# split: train, val, test

import torch
from src.inference.common_inference import  CONFIGURATION_INF

def argmax_strategy(similarity_map, best_idx, Hf, Wf):
    
    return torch.tensor([best_idx % Wf, best_idx // Wf], device=CONFIGURATION_INF["DEVICE"])
