# @title Inference

from Configurations import CONFIGURATION
from Utils import get_predictions
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import torch, cv2

# --- EVALUATION FUNCTION ---
# --- FOR EACH PAIR: LOAD SOURCE AND TARGET IMAGES ---
# --- EXTRACT DESCRIPTORS AND RESCALE COORDINATES ---
# --- USE COSINE SIMILARITY IN ORDER TO FIND THE CORRESPONDING POINT IN THE TARGET IMAGE. ---
# --- FINALLY, RETURN THE CORRECT GENERATED KEYPOINTS (USING THEIR DISTANCE FROM THE ORIGINAL ONE) RATIO. ---
# --- OPTIONALLY: VISUALIZE RESULTS. ---

def run_evaluation(loader, split_desc, visualize=False):
    total_correct = {alpha: 0.0 for alpha in CONFIGURATION["ALPHA"]}
    total_correct_keypoints = {alpha: 0 for alpha in CONFIGURATION["ALPHA"]}       # DICTS FOR PERFORMANCE
    total_images = 0
    total_keypoints = 0

    for batch in tqdm(loader, desc="Evaluating " + split_desc + " PCK metrics"):
        trg_kps = torch.as_tensor(batch["trg_kps"][0]).to(device=CONFIGURATION["DEVICE"], dtype=torch.float32)
        trg_bndbox = torch.as_tensor(batch["trg_bndbox"][0]).to(device=CONFIGURATION["DEVICE"], dtype=torch.float32)

        pred_kps = get_predictions(batch)              # PREDICTIONS
        max_dim = max(trg_bndbox[2]-trg_bndbox[0], trg_bndbox[3]-trg_bndbox[1])         # NORMALIZATION FACTOR

        image_correct_counts = {alpha: 0 for alpha in CONFIGURATION["ALPHA"]}           # FOR CURRENT IMAGE
        num_kps_in_image = len(batch["kps_ids"][0])
        total_images += 1
        total_keypoints += num_kps_in_image

        for i in range(num_kps_in_image):
            dist = torch.norm(pred_kps[i] - trg_kps[i]).item()          # DISTANCE METRIC

            for alpha in CONFIGURATION["ALPHA"]:
                if dist <= alpha * max_dim:
                    image_correct_counts[alpha] += 1
                    total_correct_keypoints[alpha] += 1            # PREDICTION IS CORRECT?

        for alpha in CONFIGURATION["ALPHA"]:
            img_accuracy = (100.0 * image_correct_counts[alpha] / num_kps_in_image)          # PCK FOR THIS IMAGE
            total_correct[alpha] += img_accuracy

        if visualize:
            visualize_keypoints(batch["src_path"][0], batch["trg_path"][0], batch["src_kps"][0],      # OPTIONAL VISUALIZATION
                                pred_kps.cpu().numpy(), trg_kps.cpu().numpy())

    print()
    print("PCK@t results per image (Average of individual image accuracies):")
    for alpha in CONFIGURATION["ALPHA"]:
        avg_pck_image = round(total_correct[alpha] / total_images, 2)
        print("PCK@" + str(alpha) + ": " +  str(avg_pck_image) + "%")

    print()
    print("PCK@t results per keypoint (Total correct / Total points):")
    for alpha in CONFIGURATION["ALPHA"]:
        avg_pck_kps = round(100.0 * total_correct_keypoints[alpha] / total_keypoints, 2)
        print("PCK@" + str(alpha) + ": " +  str(avg_pck_kps) + "%")

    return

# --- VISUALIZE RESULTS AND COMPARE CORRECT AND PREDICTED KEYPOINTS ON TARGET IMAGE. ---

def visualize_keypoints(src_path, trg_path, src_kps, pred_kps, trg_kps):
    src_img = cv2.imread(src_path)[:, :, ::-1]
    trg_img = cv2.imread(trg_path)[:, :, ::-1]

    (_, axes) = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(src_img)
    axes[0].scatter(src_kps[:,0], src_kps[:,1], c="r", s=40, label="src_kps")          # SOURCE KEYPOINTS
    axes[0].set_title("Source Image")

    axes[1].imshow(trg_img)
    axes[1].scatter(pred_kps[:,0], pred_kps[:,1], c="b", s=40, label="pred_kps")
    axes[1].scatter(trg_kps[:,0], trg_kps[:,1], c="g", s=40, marker="X", label="gt_kps")       # TARGET AND PREDICTED
    axes[1].set_title("Target Image")

    plt.legend()
    plt.show()
    return