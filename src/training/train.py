# Training functionality for semantic correspondence models
# backbone: dinov2, dinov3, sam  
# dataset: spair71k (training only supports SPair-71k)

import torch
import torch.nn.functional as F
import importlib
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.inference.common_inference import CONFIGURATION_INF
from src.inference.run_evaluation import run_evaluation
from src.inference.argmax import argmax_strategy

CONFIGURATION_TRAIN = {
    "TAU": 0.05,
    "LEARNING_RATE": 1e-5,
    "WEIGHT_DECAY": 1e-1, 
    "NUM_EPOCHS": 1,
    "NUM_LAYERS": 1,
    "BATCH_SIZE": 32,   # 8 for SAM due to memory constraints
    "PATH_BEST_MODEL": "/content/drive/MyDrive/AML-Semantic-Correspondence/weights/best_model.pth",
}

def custom_collate_fn(batch):
    """Collate function for training batches with variable-sized data."""
    collated_batch = {}
    keys = batch[0].keys()
    for key in keys:
        collated_batch[key] = [d[key] for d in batch]
    return collated_batch

def setup_light_finetuning(model, backbone):
    """
    Unfreeze only the last num_layers and the final layer norm.
    
    Args:
        model: The model to finetune
        backbone: Model backbone type ('dinov2', 'dinov3', 'sam')
    
    Returns:
        None (modifies model in-place)
    """
    N_Params = 0
    N_Free_Params = 0

    if backbone == "sam":
        blocks_to_unfreeze = model.image_encoder.blocks[-CONFIGURATION_TRAIN["NUM_LAYERS"]:]
        norm = getattr(model.image_encoder, "post_norm", None)
    else:
        blocks_to_unfreeze = model.blocks[-CONFIGURATION_TRAIN["NUM_LAYERS"]:]
        norm = model.norm

    # Freeze all parameters first
    for param in model.parameters():
        N_Params += param.numel()
        param.requires_grad = False

    # Unfreeze the last layers
    for block in blocks_to_unfreeze:
        for param in block.parameters():
            N_Free_Params += param.numel()
            param.requires_grad = True

    # Unfreeze norm if it exists
    if norm:
        for param in norm.parameters():
            N_Free_Params += param.numel()
            param.requires_grad = True

    print("Total parameters:", N_Params)
    print("Total trainable:", N_Free_Params)
    print("Percentage trainable:", round(100 * N_Free_Params / N_Params, 2), "%")

def get_split_loss(batch, backbone, model, grad=True, index=0):
    """
    Compute training loss for a single sample in the batch.
    
    Args:
        batch: Batch data from DataLoader
        backbone: Model backbone type
        model: The model to use for feature extraction
        grad: Whether to enable gradients
        index: Index of sample in batch to process
        
    Returns:
        torch.Tensor: Computed loss
    """
    get_descriptors = importlib.import_module(f"src.utils.{backbone}_utils").get_descriptors
    
    src_kps = torch.as_tensor(batch["src_kps"][index]).to(device=CONFIGURATION_INF["DEVICE"], dtype=torch.float32)
    trg_kps = torch.as_tensor(batch["trg_kps"][index]).to(device=CONFIGURATION_INF["DEVICE"], dtype=torch.float32)

    # Get descriptors (this should use the model parameter)
    (feat_src, sw, sh) = get_descriptors(batch["src_path"][index], grad, model)
    (feat_trg, tw, th) = get_descriptors(batch["trg_path"][index], grad, model)

    (_, Hf, Wf, D) = feat_trg.shape
    trg_flat = feat_trg[0].reshape(Hf * Wf, D)
    class_loss = 0

    for i in range(src_kps.shape[0]):
        sx = int(src_kps[i, 0] * Wf / sw)
        sy = int(src_kps[i, 1] * Hf / sh)
        sx = torch.clamp(torch.tensor(sx, device=CONFIGURATION_INF["DEVICE"]), 0, Wf - 1)
        sy = torch.clamp(torch.tensor(sy, device=CONFIGURATION_INF["DEVICE"]), 0, Hf - 1)

        src_desc = feat_src[0, sy, sx, :]
        sim = torch.matmul(trg_flat, src_desc)

        # Ground truth target position rescaled to feature map
        gx = int(trg_kps[i, 0] * Wf / tw)
        gy = int(trg_kps[i, 1] * Hf / th)
        gx = torch.clamp(torch.tensor(gx, device=CONFIGURATION_INF["DEVICE"]), 0, Wf - 1)
        gy = torch.clamp(torch.tensor(gy, device=CONFIGURATION_INF["DEVICE"]), 0, Hf - 1)

        gt_index = gy * Wf + gx
        class_loss = class_loss + F.cross_entropy(
            (sim / CONFIGURATION_TRAIN["TAU"]).unsqueeze(0),
            gt_index.unsqueeze(0), 
            label_smoothing=0.1
        )

    return class_loss

def get_total_loss(loader, backbone, model):
    """
    Compute total validation loss across the entire validation dataset.
    
    Args:
        loader: DataLoader for validation data
        backbone: Model backbone type
        model: The model to evaluate
        
    Returns:
        float: Mean validation loss
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating validation loss"):
            bs = len(batch["src_kps"])
            batch_loss_sum = 0.0

            for i in range(bs):
                class_loss = get_split_loss(batch, backbone, model, grad=False, index=i)
                batch_loss_sum += class_loss.item()

            total_loss += (batch_loss_sum / bs)

    return total_loss / len(loader)

def train_step(backbone):
    
    global_loss = float("inf")
    
    # Load model dynamically based on backbone
    backbone_utils = importlib.import_module(f"src.utils.{backbone}_utils")
    CONFIGURATION_MODEL = backbone_utils.CONFIGURATION_MODEL
    
    # Load model using dynamic imports based on backbone type
    if backbone == "sam":
        from segment_anything import sam_model_registry
        model = sam_model_registry["vit_b"](checkpoint=CONFIGURATION_MODEL["PTH_PATH_SAM"])
    else:
        model_name = f"{backbone}_vitb{'14_reg' if backbone == 'dinov2' else '16'}"
        model = torch.hub.load(f"facebookresearch/{backbone}", model_name, weights=CONFIGURATION_MODEL[f"PTH_PATH_{backbone.upper()}"])
    
    model = model.to(CONFIGURATION_MODEL["DEVICE"])
    
    # Create training and validation datasets on-the-fly
    dataset_module = importlib.import_module(f"src.dataset.spair71k")
    CONFIGURATION_DS = dataset_module.CONFIGURATION_DS
    
    # Training dataset
    train_dataset = dataset_module.Dataset_spair71k(
        CONFIGURATION_DS["ALL_TRAIN_PATH"], 
        CONFIGURATION_DS["PATH_TRAIN"]
    )
    loader_train = DataLoader(
        train_dataset, 
        CONFIGURATION_TRAIN["BATCH_SIZE"], 
        collate_fn=custom_collate_fn
    )
    
    # Validation dataset  
    val_dataset = dataset_module.Dataset_spair71k(
        CONFIGURATION_DS["ALL_VAL_PATH"], 
        CONFIGURATION_DS["PATH_VAL"]
    )
    loader_val = DataLoader(val_dataset, batch_size=1)

    print()
    print("="*60)
    print("PERFORMING INITIAL EVALUATION ON PRE-TRAINED MODEL")
    print("="*60)

    # Initial validation loss and PCK evaluation
    initial_val_loss = get_total_loss(loader_val, backbone, model)
    print()
    run_evaluation(backbone, "spair71k", "val", argmax_strategy)
    print("Pre-tuning Validation Loss:", initial_val_loss)
    print("="*60)
    print()

    # Setup fine-tuning
    print(f"Fine-tuning on {CONFIGURATION_TRAIN['NUM_LAYERS']} free layers")
    setup_light_finetuning(model, backbone)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, 
        lr=CONFIGURATION_TRAIN["LEARNING_RATE"], 
        weight_decay=CONFIGURATION_TRAIN["WEIGHT_DECAY"]
    )

    # Training loop
    for epoch in range(CONFIGURATION_TRAIN["NUM_EPOCHS"]):
        
        # Training phase
        model.train()
        running_train_loss = 0.0
        pbar = tqdm(loader_train, desc=f"Epoch {epoch}")

        for batch in pbar:
            optimizer.zero_grad()
            bs = len(batch["src_kps"])
            class_loss = 0.0

            for i in range(bs):
                class_loss += get_split_loss(batch, backbone, model, index=i)

            class_loss /= bs
            class_loss.backward()
            optimizer.step()
            running_train_loss += class_loss.item()
            pbar.set_postfix(loss=class_loss.item())

        epoch_train_loss = running_train_loss / len(loader_train)

        # Validation phase
        val_loss = get_total_loss(loader_val, backbone, model)

        print("="*60)
        print("Epoch train loss:", epoch_train_loss)
        print("Epoch validation loss:", val_loss)
        print()
        run_evaluation(backbone, "spair71k", "val", argmax_strategy)

        if val_loss < global_loss:
            global_loss = val_loss
            torch.save(model.state_dict(), CONFIGURATION_TRAIN["PATH_BEST_MODEL"])
            print("BEST MODEL SAVED!")

        print("="*60)
        print()
