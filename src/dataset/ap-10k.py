from torch.utils.data import Dataset
import os, json, numpy as np
import random

CONFIGURATION_DS = {
    
  "PATH_AP10K": "/content/drive/MyDrive/AML-Semantic-Correspondence/datasets/ap-10k.zip",
  "PATH_ANNOTATIONS_AP10K_TEST": "/content/ap-10k/annotations/ap10k-test-split1.json",
  "IMAGE_FOLDER_NAME_AP10K": "/content/ap-10k/data"

}

class Dataset_ap10k(Dataset):

    def __init__(self):
        # Load annotations and images
        with open(CONFIGURATION_DS['PATH_ANNOTATIONS_AP10K_TEST'], 'r') as f:
            self.data = json.load(f)
        
        self.annotations = self.data['annotations']
        
        # Create images dictionary for efficient access
        self.images = {img['id']: img for img in self.data['images']}
        
        # Group annotations by species_id (category_id in AP-10K)
        species_groups = {}
        for ann in self.annotations:
            species_id = ann['category_id']
            if species_id not in species_groups:
                species_groups[species_id] = []
            species_groups[species_id].append(ann)

        # Create pairs by shuffling and pairing neighbors within each species
        self.pairs = []
        for species_id, anns in species_groups.items():
            if len(anns) < 2:
                continue

            # Create pairs by shuffling and pairing neighbors
            shuffled = anns.copy()
            random.shuffle(shuffled)

            for i in range(len(shuffled)):
                src_ann = shuffled[i]
                trg_ann = shuffled[(i + 1) % len(shuffled)]
                self.pairs.append((src_ann, trg_ann))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ann, trg_ann = self.pairs[idx]

        # Load Source Data
        src_info = self.images[src_ann['image_id']]
        src_path = os.path.join(CONFIGURATION_DS['IMAGE_FOLDER_NAME_AP10K'], src_info['file_name'])
        src_keypoints = np.array(src_ann['keypoints']).reshape(-1, 3)

        # Load Target Data
        trg_info = self.images[trg_ann['image_id']]
        trg_path = os.path.join(CONFIGURATION_DS['IMAGE_FOLDER_NAME_AP10K'], trg_info['file_name'])
        trg_keypoints = np.array(trg_ann['keypoints']).reshape(-1, 3)

        # Find keypoints that are visible in BOTH images
        # visibility > 0 means the point exists and is labeled
        common_mask = (src_keypoints[:, 2] > 0) & (trg_keypoints[:, 2] > 0)

        src_kps = src_keypoints[common_mask, :2].astype(np.float32)
        trg_kps = trg_keypoints[common_mask, :2].astype(np.float32)
        kps_ids = np.where(common_mask)[0].astype(np.int32)

        # Target Bounding Box for PCK normalization
        bbox = np.array(trg_ann['bbox'], dtype=np.float32)  # [x, y, width, height]
        trg_bndbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)

        return {
            "src_path": src_path,
            "trg_path": trg_path,
            "src_kps": src_kps,
            "trg_kps": trg_kps,
            "kps_ids": kps_ids,
            "trg_bndbox": trg_bndbox,
        }