
from torch.utils.data import Dataset
import numpy as np

CONFIGURATION_DS = {
    
  "PATH_WILLOW": "/content/drive/MyDrive/AML-Semantic-Correspondence/datasets/PF-dataset.zip",
  "PATH_ANNOTATIONS_WILLOW": "/content/drive/MyDrive/AML-Semantic-Correspondence/datasets/test_pairs.txt",
  "LEN": 10

}

class Dataset_pf_willow(Dataset):

    def __init__(self):
        self.pair_files = []
        file = open(CONFIGURATION_DS["PATH_ANNOTATIONS_WILLOW"], "r")
        self.pair_files = file.readlines()[1:]                              # TAKE ALL FILES
        file.close()
        return

    def __len__(self):
        return len(self.pair_files)


    def __getitem__(self, idx):
       ann = self.pair_files[idx].strip().split(",")
       src_kps = []
       trg_kps = []

       # SRC_KPS

       for i in range(2, len(ann)-3*CONFIGURATION_DS["LEN"]):
           src_kps.append((ann[i], ann[i+CONFIGURATION_DS["LEN"]]))

       # TRG_KPS

       for i in range(2+2*CONFIGURATION_DS["LEN"], len(ann)-CONFIGURATION_DS["LEN"]):    # TRG_KPS
           trg_kps.append((ann[i], ann[i+CONFIGURATION_DS["LEN"]]))

       src_kps = np.array(src_kps, dtype=np.float32)
       trg_kps = np.array(trg_kps, dtype=np.float32)

       # BBOX

       x_min = np.min(trg_kps[:,0])
       y_min = np.min(trg_kps[:,1])           
       x_max = np.max(trg_kps[:,0])
       y_max = np.max(trg_kps[:,1])

       trg_bndbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

       return {
          "src_path": ann[0],
          "trg_path": ann[1],
          "src_kps": src_kps,                   
          "trg_kps": trg_kps,
          "trg_bndbox": trg_bndbox,
          "kps_ids": np.arange(CONFIGURATION_DS["LEN"], dtype=np.int32),
          }
