# @title Utils

from Configurations import CONFIGURATION, PREPROCESS, MODEL
from PIL import Image
import torch, torch.nn.functional as F, math

# --- HELPER FUNCTIONS ---
# --- LOAD AND PREPROCESS IMAGE ---
# --- RESHAPE AND NORMALIZE FEATURE MAPS ---
# --- FINALLY RETURN THE EXTRACTED FEATURE MAPS AND THE ORIGINAL DIMENSIONS ---

def get_descriptors(img_path, grad):
    img = Image.open(img_path).convert("RGB")
    (w, h) = img.size
    input_tensor = PREPROCESS(img).unsqueeze(0).to(CONFIGURATION["DEVICE"])

    with torch.set_grad_enabled(grad):
        if CONFIGURATION["MODEL_VERSION"] == "dinov2":
            x = MODEL.get_intermediate_layers(input_tensor, n=1)[0]   # [B, N, D]
            (B, N, D) = x.shape
            H = int(math.sqrt(N))
            x = x.reshape(B, H, H, D)                # RESHAPE

        elif CONFIGURATION["MODEL_VERSION"] == "dinov3":
            x = MODEL.forward_features(input_tensor)["x_norm_patchtokens"]
            (B, N, D) = x.shape
            H = int(math.sqrt(N))
            x = x.reshape(B, H, H, D)               # RESHAPE

        elif CONFIGURATION["MODEL_VERSION"] == "sam":
            x = MODEL.image_encoder(input_tensor).permute(0, 2, 3, 1)

    x = F.normalize(x, dim=-1)
    return (x, w, h)

# --- GET PREDICTIONS ---
# --- EXTRACT DESCRIPTORS, FOR EACH SRC_KPS RESCALE IT ---
# --- USE COSINE SIMILARITY METRIC AND COMPUTE THE PREDICTION WITH SIMPLE ARGMAX  ---

def get_predictions(batch, index=0):
    src_kps = batch["src_kps"][index]

    (feat_src, sw, sh) = get_descriptors(batch["src_path"][index], False)
    (feat_trg, tw, th) = get_descriptors(batch["trg_path"][index], False)             # DESCRIPTORS

    (_, Hf, Wf, D) = feat_trg.shape
    trg_flat = feat_trg[0].reshape(Hf * Wf, D)

    pred_kps = []

    for i in range(src_kps.shape[0]):
        sx = int(src_kps[i, 0] * Wf / sw)
        sy = int(src_kps[i, 1] * Hf / sh)
        sx = torch.clamp(torch.tensor(sx, device=CONFIGURATION["DEVICE"]), 0, Wf - 1)              # RESIZE
        sy = torch.clamp(torch.tensor(sy, device=CONFIGURATION["DEVICE"]), 0, Hf - 1)

        src_desc = feat_src[0, sy, sx, :]
        sim = torch.matmul(trg_flat, src_desc)            # COSINE SIMILARITY

        best_idx = sim.argmax()
        pred_xy = torch.tensor([best_idx % Wf, best_idx // Wf], device=CONFIGURATION["DEVICE"])           # PREDICTION

        pred_x = (pred_xy[0] + 0.5) * (tw / Wf)
        pred_y = (pred_xy[1] + 0.5) * (th / Hf)
        pred_kps.append(torch.stack([pred_x, pred_y]))

    return torch.stack(pred_kps)