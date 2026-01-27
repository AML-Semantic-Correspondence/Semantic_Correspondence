
from torch.utils.data import Dataset
import os, json, numpy as np

CONFIGURATION_DS = {

    "PATH_FILE": "/content/drive/MyDrive/AML-Semantic-Correspondence/datasets/SPair-71k.tar.gz",
    "PATH_TEST": "/content/SPair-71k/PairAnnotation/test",
    "PATH_VAL": "/content/SPair-71k/PairAnnotation/val",
    "PATH_TRAIN": "/content/SPair-71k/PairAnnotation/trn",
    "ALL_TEST_PATH": "/content/SPair-71k/Layout/large/test.txt",
    "ALL_TRAIN_PATH": "/content/SPair-71k/Layout/large/trn.txt",
    "ALL_VAL_PATH": "/content/SPair-71k/Layout/large/val.txt",
    "IMAGE_FOLDER_NAME": "/content/SPair-71k/JPEGImages",

}

class Dataset_spair71k(Dataset):

    def __init__(self, pair_path, source_path):
        self.image_path = source_path
        file = open(pair_path, "r")
        self.pair_files = file.readlines()                
        file.close()
        return

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, idx):
        file_name = self.pair_files[idx].strip()
        category = file_name.split(".json")[0].split(":")[1]
        json_path = os.path.join(self.image_path, file_name + ".json")

        file = open(json_path, "r")
        annotation = json.load(file)             
        file.close()

        src_path = os.path.join(CONFIGURATION_DS["IMAGE_FOLDER_NAME"], category, annotation["src_imname"])
        trg_path = os.path.join(CONFIGURATION_DS["IMAGE_FOLDER_NAME"], category, annotation["trg_imname"])
        ids = [int(el) for el in annotation["kps_ids"]]

        return {
            "src_path": src_path,
            "trg_path": trg_path,
            "src_kps": np.array(annotation["src_kps"]),
            "trg_kps": np.array(annotation["trg_kps"]),
            "kps_ids": np.array(ids),
            "trg_bndbox": np.array(annotation["trg_bndbox"]),   
        }