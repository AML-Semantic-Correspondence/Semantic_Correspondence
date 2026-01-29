# Common inference functionality shared across different prediction methods
# backbone: dinov2, dinov3, sam
# dataset: spair71k, pf-pascal, pf-willow
# split: train, val, test

import importlib, torch
from torch.utils.data import DataLoader

CONFIGURATION_INF = {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "ALPHA": [0.05, 0.1, 0.2],
}

def get_predictions_from_batch(batch, backbone, prediction_method, model):
   
    get_descriptors = importlib.import_module(f"src.utils.{backbone}_utils").get_descriptors
    
    src_kps = batch["src_kps"][0]
    
    (feat_src, sw, sh) = get_descriptors(batch["src_path"][0], False, model)
    (feat_trg, tw, th) = get_descriptors(batch["trg_path"][0], False, model)
    
    # Prepare feature maps
    (_, Hf, Wf, D) = feat_trg.shape
    trg_flat = feat_trg[0].reshape(Hf * Wf, D)
    
    pred_kps = []
    
    # Process each keypoint
    for i in range(src_kps.shape[0]):
        # Rescale source keypoint coordinates to feature map dimensions
        sx = int(src_kps[i, 0] * Wf / sw)
        sy = int(src_kps[i, 1] * Hf / sh)
        sx = torch.clamp(torch.tensor(sx, device=CONFIGURATION_INF["DEVICE"]), 0, Wf - 1)
        sy = torch.clamp(torch.tensor(sy, device=CONFIGURATION_INF["DEVICE"]), 0, Hf - 1)
        
        # Extract source descriptor and compute similarity
        src_desc = feat_src[0, sy, sx, :]
        similarity_map = torch.matmul(trg_flat, src_desc)
        best_idx = similarity_map.argmax()
        
        # Apply specific prediction method
        pred_xy = prediction_method(similarity_map, best_idx, Hf, Wf)
        
        # Convert feature map coordinates back to image coordinates
        pred_x = (pred_xy[0] + 0.5) * (tw / Wf)
        pred_y = (pred_xy[1] + 0.5) * (th / Hf)
        pred_kps.append(torch.stack([pred_x, pred_y]))
    
    return torch.stack(pred_kps)
