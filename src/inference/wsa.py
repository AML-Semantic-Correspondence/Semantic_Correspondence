
# backbone: dinov2, dinov3, sam
# dataset: spair71k, pf-pascal, pf-willow
# split: train, val, test

import torch
import torch.nn.functional as F
from src.inference.common_inference import CONFIGURATION_INF

CONFIGURATION_WSA = {
    "WINDOW_SOFTMAX": 5,
    "TAU_SOFTMAX": 0.05
}

def wsa_strategy(similarity_map, best_idx, Hf, Wf):
    
    similarity_map = similarity_map.view(Hf, Wf)
    y_peak = best_idx // Wf
    x_peak = best_idx % Wf

    half = CONFIGURATION_WSA["WINDOW_SOFTMAX"] // 2
    y0 = max(y_peak - half, 0)
    y1 = min(y_peak + half + 1, Hf)
    x0 = max(x_peak - half, 0)
    x1 = min(x_peak + half + 1, Wf)
    window = similarity_map[y0:y1, x0:x1]

    (ys, xs) = torch.meshgrid(
        torch.arange(y0, y1, device=similarity_map.device),
        torch.arange(x0, x1, device=similarity_map.device),
        indexing="ij")

    coords = torch.stack([xs.flatten(), ys.flatten()], dim=1).float()
    prob = F.softmax(window.flatten() / CONFIGURATION_WSA["TAU_SOFTMAX"], dim=0)

    return (coords * prob[:, None]).sum(dim=0)