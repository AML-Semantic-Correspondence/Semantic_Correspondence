# @title Datasets

from torch.utils.data import Dataset, DataLoader
from Configurations import CONFIGURATION
import os, json, numpy as np

# --- DATASET CLASSES ---
# --- COLLECT ALL JSON FILE BASED ---
# --- WHEN REQUIRED, OPEN THE NEXT FILE AND TAKE IN A DICTIONARY ALL YOU NEED ---

class SPair71kDataset(Dataset):
    def __init__(self, pair_path, source_path, folder_path):
        self.image_path = source_path
        self.folder_path = folder_path
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

        src_path = os.path.join(self.folder_path, category, annotation["src_imname"])
        trg_path = os.path.join(self.folder_path, category, annotation["trg_imname"])
        ids = [int(el) for el in annotation["kps_ids"]]

        return {
            "src_path": src_path,
            "trg_path": trg_path,
            "src_kps": np.array(annotation["src_kps"]),
            "trg_kps": np.array(annotation["trg_kps"]),
            "kps_ids": np.array(ids),
            "trg_bndbox": np.array(annotation["trg_bndbox"]),        # DICT of BATCHES (SIZE=1)
        }

def custom_collate_fn(batch):
    collated_batch = {}
    keys = batch[0].keys()
    for key in keys:
        collated_batch[key] = [d[key] for d in batch]              # COLLATE FOR DATALOADER
    return collated_batch

# --- GENERATE DATALOADER FROM REQUIRED OPERATION --

def Loader(index, batch_size):
    if index == 0:                                         # FOR INFERENCE WITH SPAIR
        dataset=SPair71kDataset(CONFIGURATION["ALL_TEST_PATH_SPAIR71K"], CONFIGURATION["PATH_TEST_SPAIR71K"],
                                CONFIGURATION["IMAGE_FOLDER_NAME_SPAIR71K"])

    if batch_size>1:
        loader=DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    else:
        loader=DataLoader(dataset, batch_size=1)

    return loader