# @title Datasets

from torch.utils.data import Dataset, DataLoader
from Configurations import CONFIGURATION
import os, json, numpy as np, scipy.io as sio

# --- DATASET CLASSES ---
# --- COLLECT ALL JSON FILE ---
# --- WHEN REQUIRED, OPEN THE NEXT FILE AND TAKE IN A DICTIONARY ALL YOU NEED ---

class SPair71kDataset(Dataset):

    def __init__(self, pair_path, source_path):
        self.image_path = source_path
        file = open(pair_path, "r")
        self.pair_files = file.readlines()                # TAKE ALL JSON FILES
        file.close()
        return

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, idx):
        file_name = self.pair_files[idx].strip()
        category = file_name.split(".json")[0].split(":")[1]
        json_path = os.path.join(self.image_path, file_name + ".json")

        file = open(json_path, "r")
        annotation = json.load(file)             # LOAD INFORMATIONS
        file.close()

        src_path = os.path.join(CONFIGURATION["IMAGE_FOLDER_NAME_SPAIR71K"], category, annotation["src_imname"])
        trg_path = os.path.join(CONFIGURATION["IMAGE_FOLDER_NAME_SPAIR71K"], category, annotation["trg_imname"])
        ids = [int(el) for el in annotation["kps_ids"]]

        return {
            "src_path": src_path,
            "trg_path": trg_path,
            "src_kps": np.array(annotation["src_kps"]),
            "trg_kps": np.array(annotation["trg_kps"]),
            "kps_ids": np.array(ids),
            "trg_bndbox": np.array(annotation["trg_bndbox"]),        # DICT of BATCHES (SIZE=1)
        }

class PFPascalDataset(Dataset):

    def __init__(self):
        self.pair_files = []
        dir = os.listdir(CONFIGURATION["PATH_ANNOTATIONS_PASCAL"])

        for sottodir in dir:
            if sottodir.startswith("."): continue         # SHALLOWED DIRECTORIES

            for file in os.listdir(os.path.join(CONFIGURATION["PATH_ANNOTATIONS_PASCAL"], sottodir)):
                self.pair_files.append((sottodir,file.split(".jpg")[0]))                # TAKE ALL FILES
        return

    def __len__(self):
         return len(self.pair_files)

    def __getitem__(self, idx):
       (src_name, trg_name) = self.pair_files[idx][1].split("-")              # (CATEGORY,ID)

       src_ann = os.path.join(CONFIGURATION["ALL_ANNOTATIONS_PASCAL"], self.pair_files[idx][0], src_name + ".mat")
       trg_ann = os.path.join(CONFIGURATION["ALL_ANNOTATIONS_PASCAL"], self.pair_files[idx][0], trg_name + ".mat")     # ANNOTATIONS

       ann = sio.loadmat(src_ann)
       ann2 = sio.loadmat(trg_ann)               # ANNOTATIONS

       src_kps = np.array(ann["kps"])
       trg_kps = np.array(ann2["kps"])
       valid_mask = ~np.isnan(src_kps).any(axis=1) & (~np.isnan(trg_kps).any(axis=1))    # VALID KEYPOINTS

       return {
          "src_path": os.path.join(CONFIGURATION["ALL_IMAGES_PASCAL"], src_name + ".jpg"),
          "trg_path": os.path.join(CONFIGURATION["ALL_IMAGES_PASCAL"], trg_name + ".jpg"),
          "src_kps": src_kps[valid_mask],                   # GET DATA
          "trg_kps": trg_kps[valid_mask],
          "trg_bndbox": np.array(ann2["bbox"]).squeeze(),
          "kps_ids": np.arange(len(ann["kps"][valid_mask]), dtype=np.int32),
          }

def custom_collate_fn(batch):
    collated_batch = {}
    keys = batch[0].keys()
    for key in keys:
        collated_batch[key] = [d[key] for d in batch]              # COLLATE FOR DATALOADER
    return collated_batch

# --- GENERATE DATALOADER FROM REQUIRED OPERATION --

def Loader(index):            # TEST, TRAINING OR EVALUATION
    if index == 0:
        dataset = SPair71kDataset(CONFIGURATION["ALL_TEST_PATH_SPAIR71K"], CONFIGURATION["PATH_TEST_SPAIR71K"])
        loader = DataLoader(dataset, 1)

    elif index == 1:
        dataset = SPair71kDataset(CONFIGURATION["ALL_TRAIN_PATH_SPAIR71K"], CONFIGURATION["PATH_TRAIN_SPAIR71K"])
        loader = DataLoader(dataset, CONFIGURATION["BATCH_SIZE"], collate_fn=custom_collate_fn)

    elif index == 2:
        dataset = SPair71kDataset(CONFIGURATION["ALL_VAL_PATH_SPAIR71K"], CONFIGURATION["PATH_VAL_SPAIR71K"])
        loader = DataLoader(dataset, 1)

    elif index == 3:
        dataset = PFPascalDataset()
        loader = DataLoader(dataset, 1)                   # INFERENCE WITH PASCAL

    return loader