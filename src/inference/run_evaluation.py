from tqdm.auto import tqdm
import torch
import importlib
from torch.utils.data import DataLoader

from .common_inference import get_predictions_from_batch, CONFIGURATION_INF


def run_evaluation(backbone, dataset_var, split, prediction_method, weights_path=None):
    """
    Run evaluation on a dataset using specified prediction method.
    
    Args:
        backbone: Model backbone ('dinov2', 'dinov3', 'sam')
        dataset_var: Dataset to use ('spair71k', 'pf-pascal', 'pf-willow')
        split: Data split to use ('train', 'val', 'test')
        prediction_method: Function that takes (similarity_map, best_idx, Hf, Wf)
    
    Returns:
        None (prints results)
    """
    # Load model dynamically based on backbone
    backbone_utils = importlib.import_module(f"..utils.{backbone}_utils")
    CONFIGURATION_MODEL = backbone_utils.CONFIGURATION_MODEL
    
    # Load model using dynamic imports based on backbone type
    if backbone == "sam":
        from segment_anything import sam_model_registry
        model = sam_model_registry["vit_b"](checkpoint=weights_path if weights_path else CONFIGURATION_MODEL["PTH_PATH_SAM"])
    else:
        model_name = f"{backbone}_vitb{'14_reg' if backbone == 'dinov2' else '16'}"
        model = torch.hub.load(f"facebookresearch/{backbone}", model_name, weights=weights_path if weights_path else CONFIGURATION_MODEL[f"PTH_PATH_{backbone.upper()}"])
    
    model = model.to(CONFIGURATION_MODEL["DEVICE"])
    model.eval()
    
    # Create dataset and loader on-the-fly
    dataset_module = importlib.import_module(f"..dataset.{dataset_var}")
    CONFIGURATION_DS = dataset_module.CONFIGURATION_DS
    split = split.upper()
    
    match dataset_var:
        case "spair71k":
            dataset = dataset_module.Dataset_spair71k(
                CONFIGURATION_DS[f"ALL_{split}_PATH"], 
                CONFIGURATION_DS[f"PATH_{split}"]
            )
        case _:
            class_name = f"Dataset_{dataset_var.replace('-', '_')}"
            dataset = getattr(dataset_module, class_name)()
    
    loader = DataLoader(dataset, batch_size=1)
    
    total_correct = {alpha: 0.0 for alpha in CONFIGURATION_INF["ALPHA"]}
    total_correct_keypoints = {alpha: 0 for alpha in CONFIGURATION_INF["ALPHA"]}      
    total_images = 0
    total_keypoints = 0

    for batch in tqdm(loader, desc=f"Evaluating {dataset_var} {split} PCK metrics"):
        trg_kps = torch.as_tensor(batch["trg_kps"][0]).to(device=CONFIGURATION_INF["DEVICE"], dtype=torch.float32)
        trg_bndbox = torch.as_tensor(batch["trg_bndbox"][0]).to(device=CONFIGURATION_INF["DEVICE"], dtype=torch.float32)

        pred_kps = get_predictions_from_batch(batch, backbone, prediction_method, model)              # PREDICTIONS
        max_dim = max(trg_bndbox[2]-trg_bndbox[0], trg_bndbox[3]-trg_bndbox[1])         # NORMALIZATION FACTOR

        image_correct_counts = {alpha: 0 for alpha in CONFIGURATION_INF["ALPHA"]}           # FOR CURRENT IMAGE
        num_kps_in_image = len(batch["kps_ids"][0])
        total_images += 1
        total_keypoints += num_kps_in_image

        for i in range(num_kps_in_image):
            dist = torch.norm(pred_kps[i] - trg_kps[i]).item()          # DISTANCE METRIC

            for alpha in CONFIGURATION_INF["ALPHA"]:
                if dist <= alpha * max_dim:
                    image_correct_counts[alpha] += 1
                    total_correct_keypoints[alpha] += 1            # PREDICTION IS CORRECT?

        for alpha in CONFIGURATION_INF["ALPHA"]:
            img_accuracy = (100.0 * image_correct_counts[alpha] / num_kps_in_image)          # PCK FOR THIS IMAGE
            total_correct[alpha] += img_accuracy
    print()
    print("PCK@t results per image (Average of individual image accuracies):")
    for alpha in CONFIGURATION_INF["ALPHA"]:
        avg_pck_image = round(total_correct[alpha] / total_images, 2)
        print("PCK@" + str(alpha) + ": " +  str(avg_pck_image) + "%")

    print()
    print("PCK@t results per keypoint (Total correct / Total points):")
    for alpha in CONFIGURATION_INF["ALPHA"]:
        avg_pck_kps = round(100.0 * total_correct_keypoints[alpha] / total_keypoints, 2)
        print("PCK@" + str(alpha) + ": " +  str(avg_pck_kps) + "%")

    return
