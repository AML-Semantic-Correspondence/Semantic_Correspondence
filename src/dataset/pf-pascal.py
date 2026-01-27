
from torch.utils.data import Dataset
import os, numpy as np, scipy.io as sio

CONFIGURATION_DS = {

  "PATH_PASCAL": "/content/drive/MyDrive/AML-Semantic-Correspondence/datasets/PF-dataset-PASCAL.zip",
  "PATH_ANNOTATIONS_PASCAL": "/content/PF-dataset-PASCAL/ShowMatchingPairs",
  "ALL_ANNOTATIONS_PASCAL": "/content/PF-dataset-PASCAL/Annotations",
  "ALL_IMAGES_PASCAL": "/content/PF-dataset-PASCAL/JPEGImages"
  
  }

class Dataset_pf_pascal(Dataset):

    def __init__(self):
        self.pair_files = []
        dir = os.listdir(CONFIGURATION_DS["PATH_ANNOTATIONS_PASCAL"])

        for sottodir in dir:
            if sottodir.startswith("."): continue         # SHALLOWED DIRECTORIES

            for file in os.listdir(os.path.join(CONFIGURATION_DS["PATH_ANNOTATIONS_PASCAL"], sottodir)):
                self.pair_files.append((sottodir,file.split(".jpg")[0]))                # TAKE ALL FILES
        return

    def __len__(self):
         return len(self.pair_files)

    def __getitem__(self, idx):
       (src_name, trg_name) = self.pair_files[idx][1].split("-")              # (CATEGORY,ID)

       src_ann = os.path.join(CONFIGURATION_DS["ALL_ANNOTATIONS_PASCAL"], self.pair_files[idx][0], src_name + ".mat")
       trg_ann = os.path.join(CONFIGURATION_DS["ALL_ANNOTATIONS_PASCAL"], self.pair_files[idx][0], trg_name + ".mat")     # ANNOTATIONS

       ann = sio.loadmat(src_ann)
       ann2 = sio.loadmat(trg_ann)               # ANNOTATIONS

       src_kps = np.array(ann["kps"])
       trg_kps = np.array(ann2["kps"])
       valid_mask = ~np.isnan(src_kps).any(axis=1) & (~np.isnan(trg_kps).any(axis=1))    # VALID KEYPOINTS

       return {
          "src_path": os.path.join(CONFIGURATION_DS["ALL_IMAGES_PASCAL"], src_name + ".jpg"),
          "trg_path": os.path.join(CONFIGURATION_DS["ALL_IMAGES_PASCAL"], trg_name + ".jpg"),
          "src_kps": src_kps[valid_mask],                   # GET DATA
          "trg_kps": trg_kps[valid_mask],
          "trg_bndbox": np.array(ann2["bbox"]).squeeze(),
          "kps_ids": np.arange(len(ann["kps"][valid_mask]), dtype=np.int32),
          }