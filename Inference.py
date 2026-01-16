# @title Inference

from Configurations import CONFIGURATION
from Utils import get_predictions, visualize_keypoints
from tqdm.auto import tqdm
import torch

# --- EVALUATION FUNCTION ---
# --- FOR EACH PAIR: LOAD SOURCE AND TARGET IMAGES ---
# --- EXTRACT DESCRIPTORS AND RESCALE COORDINATES ---
# --- USE COSINE SIMILARITY IN ORDER TO FIND THE CORRESPONDING POINT IN THE TARGET IMAGE. ---
# --- FINALLY, RETURN THE CORRECT GENERATED KEYPOINTS (USING THEIR DISTANCE FROM THE ORIGINAL ONE) RATIO. ---
# --- OPTIONALLY: VISUALIZE RESULTS. ---

def run_evaluation(loader, split_desc, visualize=False):
    total_correct = {alpha: 0 for alpha in CONFIGURATION["ALPHA"]}
    total_images = 0
    total_correct_keypoints = {alpha: 0 for alpha in CONFIGURATION["ALPHA"]}
    total_keypoints = 0

    for batch in tqdm(loader, desc="Evaluating " + split_desc + " PCK metrics"):
        trg_kps = torch.as_tensor(batch["trg_kps"][0]).to(device=CONFIGURATION["DEVICE"], dtype=torch.float32)
        trg_bndbox = torch.as_tensor(batch["trg_bndbox"][0]).to(device=CONFIGURATION["DEVICE"], dtype=torch.float32)

        pred_kps = get_predictions(batch)

        max_dim = max(trg_bndbox[2]-trg_bndbox[0], trg_bndbox[3]-trg_bndbox[1])
        total_correct_image = {alpha: 0 for alpha in CONFIGURATION["ALPHA"]}    # KEYPOINTS CORRECTLY CLASSIFIED
        total_points_image = 0                                      # FOR THE CURRENT IMAGE
        total_images += 1

        for i in range(len(batch["kps_ids"][0])):
            total_keypoints += 1

            dist = torch.norm(pred_kps[i] - trg_kps[i]).item()            # DISTANCE METRIC
            total_points_image += 1

            for alpha in CONFIGURATION["ALPHA"]:

                if dist <= alpha * max_dim:                     # PREDICTION IS CORRECT?
                    total_correct[alpha] += 1
                    total_correct_image[alpha] += 1

        # PRINT PER IMAGE

        for alpha in CONFIGURATION["ALPHA"]:
            total_correct_keypoints[alpha] += 100 * total_correct_image[alpha]
            total_correct_image[alpha] = round(100 * total_correct_image[alpha] / total_points_image, 2)
            total_correct[alpha] += total_correct_image[alpha]

        if visualize:
            visualize_keypoints(batch["src_path"][0], batch["trg_path"][0], batch["src_kps"][0], pred_kps,trg_kps)

    print("PCK@t results per image:")
    for alpha in CONFIGURATION["ALPHA"]:
        total_correct[alpha] = round(total_correct[alpha] / total_images, 2)
        total_correct_keypoints[alpha] = round(total_correct_keypoints[alpha] / total_keypoints, 2)
        print("PCK@" + str(alpha) + ": " + str(total_correct[alpha]) + "%")

    print()
    print("PCK@t results per keypoint:")
    for alpha in CONFIGURATION["ALPHA"]:
        print("PCK@" + str(alpha) + ": " + str(total_correct_keypoints[alpha]) + "%")

    return