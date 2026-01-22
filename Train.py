# @title Train

from Configurations import CONFIGURATION, Loader, MODEL
from Utils import get_descriptors
from Inference import run_evaluation
from tqdm.auto import tqdm
import torch, torch.nn.functional as F

# --- UNFREEZE ONLY THE LAST num_last_blocks LAYERS AND THE FINAL LAYER NORM ---
# --- FREEZE ALL LAYERS AND THEN FREE THE LASTS (IF ACCESSIBLE) ---

def setup_light_finetuning():
    N_Params = 0
    N_Free_Params = 0

    if CONFIGURATION["MODEL_VERSION"] == "sam":
        blocks_to_unfreeze = MODEL.image_encoder.blocks[-CONFIGURATION["NUM_LAYERS"]:]

        if hasattr(MODEL.image_encoder, "post_norm"):       # NOT SURE THE FINAL NORM IS ACCESSIBLE
            norm = MODEL.image_encoder.post_norm
        else:
            norm = None

    else:
        blocks_to_unfreeze = MODEL.blocks[-CONFIGURATION["NUM_LAYERS"]:]
        norm = MODEL.norm

    for param in MODEL.parameters():                # FREEZE ALL
        N_Params += param.numel()
        param.requires_grad = False

    for block in blocks_to_unfreeze:

        for param in block.parameters():
            N_Free_Params += param.numel()           # UNFREEZE
            param.requires_grad = True

    if norm:
        for param in norm.parameters():          # UNFREEZE NORM
            N_Free_Params += param.numel()
            param.requires_grad = True

    # NUMBERS

    print("Total parameters:", N_Params)
    print("Total trainable:", N_Free_Params)
    print("Percentage trainable:", round(100 * N_Free_Params / N_Params, 2), "%")
    return

# --- EXTRACT FEATURE SIZE, FOR EACH SRC_KPS RESCALE IT, USE COSINE SIMILARITY METRIC ---
# --- AND COMPUTE THE PREDICTION WITH SIMPLE ARGMAX ---

def get_split_loss(batch, grad=True, index=0):
    src_kps = torch.as_tensor(batch["src_kps"][index]).to(device=CONFIGURATION["DEVICE"], dtype=torch.float32)
    trg_kps = torch.as_tensor(batch["trg_kps"][index]).to(device=CONFIGURATION["DEVICE"], dtype=torch.float32)

    (feat_src, sw, sh) = get_descriptors(batch["src_path"][index], grad)
    (feat_trg, tw, th) = get_descriptors(batch["trg_path"][index], grad)            # GET DESCRIPTORS

    (_, Hf, Wf, D) = feat_trg.shape
    trg_flat = feat_trg[0].reshape(Hf * Wf, D)
    class_loss = 0

    for i in range(src_kps.shape[0]):
        sx = int(src_kps[i, 0] * Wf / sw)
        sy = int(src_kps[i, 1] * Hf / sh)
        sx = torch.clamp(torch.tensor(sx, device=CONFIGURATION["DEVICE"]), 0, Wf - 1)       # FOR EACH KEYPOINT
        sy = torch.clamp(torch.tensor(sy, device=CONFIGURATION["DEVICE"]), 0, Hf - 1)

        src_desc = feat_src[0, sy, sx, :]
        sim = torch.matmul(trg_flat, src_desc)            # COSINE SIMILARITY

        # --- GROUND TRUTH TARGET POSITION RESCALED TO FEATURE MAP ---

        gx = int(trg_kps[i, 0] * Wf / tw)
        gy = int(trg_kps[i, 1] * Hf / th)
        gx = torch.clamp(torch.tensor(gx, device=CONFIGURATION["DEVICE"]), 0, Wf - 1)
        gy = torch.clamp(torch.tensor(gy, device=CONFIGURATION["DEVICE"]), 0, Hf - 1)

        gt_index = gy * Wf + gx
        class_loss = class_loss + F.cross_entropy((sim / CONFIGURATION["TAU"]).unsqueeze(0),
                                                  gt_index.unsqueeze(0), label_smoothing=0.1)       # COMPUTE LOSS

    return class_loss

# --- TRAINING FUNCTION ---
# --- FIRST EVALUATE PERFORMANCE WITHOUT FINETUNING, THEN UNFREEZE LAST LAYERS ---
# --- FOR EACH BATCH, COMPUTE CROSS ENTROPY LOSS AND BACK PROPAGATE ---
# --- FINALLY SHOW THE VALUES ---

def Train_step():
    global_loss = float("inf")
    loader_train = Loader(index=1)
    loader_val = Loader(index=2)

    print()
    print("="*60)
    print("PERFORMING INITIAL EVALUATION ON PRE-TRAINED MODEL")                # INITIAL LOSS AND VALIDATION PCKS
    print("="*60)

    initial_val_loss = get_total_loss(loader_val)
    print()
    run_evaluation(loader_val, "validation")
    print("Pre-tuning Validation Loss: " + str(initial_val_loss))
    print("="*60)
    print()

    print()
    print("Fine-tuning on " + str(CONFIGURATION["NUM_LAYERS"]) + " free layers")        # UNFREEZE
    setup_light_finetuning()
    params = [p for p in MODEL.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=CONFIGURATION["LEARNING_RATE"], weight_decay=CONFIGURATION["WEIGHT_DECAY"])

    for epoch in range(CONFIGURATION["NUM_EPOCHS"]):

        # --- TRAINING PHASE ---

        MODEL.train()
        running_train_loss = 0.0
        pbar = tqdm(loader_train, desc="Epoch " + str(epoch))

        for batch in pbar:
            optimizer.zero_grad()
            bs = len(batch["src_kps"])                # BATCH LOSS TO BACK PROPAGATE
            class_loss = 0.0

            for i in range(bs): class_loss += get_split_loss(batch, index=i)

            class_loss /= bs
            class_loss.backward()
            optimizer.step()
            running_train_loss += class_loss.item()
            pbar.set_postfix(loss=class_loss.item())                    # FINAL EPOCH LOSS

        epoch_train_loss = running_train_loss / len(loader_train)

        # --- VALIDATION PHASE ---

        val_loss = get_total_loss(loader_val)

        print("="*60)
        print("Epoch train loss: " + str(epoch_train_loss))
        print("Epoch validation loss: " + str(val_loss))                  # VALIDATION LOSS AND PCKS
        print()
        run_evaluation(loader_val, "validation")

        if val_loss < global_loss:
            global_loss = val_loss
            torch.save(MODEL.state_dict(), CONFIGURATION["PATH_BEST_MODEL"])           # SAVE MODEL
            print("BEST MODEL SAVED!")

        print("="*60)
        print()

    return

# --- COMPUTE CROSS ENTROPY LOSS ON EVALUATION DATASET ---

def get_total_loss(loader):
    MODEL.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating validation loss"):
            bs = len(batch["src_kps"])
            batch_loss_sum = 0.0

            for i in range(bs):
                class_loss = get_split_loss(batch, grad=False, index=i)           # BATCH LOSS
                batch_loss_sum += class_loss.item()

            total_loss += (batch_loss_sum / bs)          # MEAN

    mean_loss = total_loss / len(loader)           # MEAN
    return mean_loss
