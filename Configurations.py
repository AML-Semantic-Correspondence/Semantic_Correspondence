# @title Configurations

from torchvision import transforms
import torch

# --- CONFIGURATIONS: SET PATHS, DEVICE, MODEL AND HYPERPARAMETERS ---

CONFIGURATION = {
	"PATH_DRIVE": "/content/drive",
	"PATH_EXPORT": "/content/drive/MyDrive/AML-Semantic-Correspondence/weights/",
	"PATH_FILE": "/content/drive/MyDrive/AML-Semantic-Correspondence/datasets/SPair-71k.tar.gz",

	"PTH_PATH": "/content/drive/MyDrive/AML-Semantic-Correspondence/weights/dinov2_vitb14_reg4_pretrain.pth",
  # "PTH_PATH": "/content/drive/MyDrive/AML-Semantic-Correspondence/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
  # "PTH_PATH": "/content/drive/MyDrive/AML-Semantic-Correspondence/weights/sam_vit_b_01ec64.pth",

  "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
  "MODEL_VERSION": "dinov2",                                        # "dinov2", "dinov3", "sam"

  # DATASET SPAIR-71k

  "PATH_TEST_SPAIR71K": "/content/SPair-71k/PairAnnotation/test",
  "PATH_VAL_SPAIR71K": "/content/SPair-71k/PairAnnotation/val",
  "PATH_TRAIN_SPAIR71K": "/content/SPair-71k/PairAnnotation/trn",

  "ALL_TEST_PATH_SPAIR71K": "/content/SPair-71k/Layout/small/test.txt",
  "ALL_TRAIN_PATH_SPAIR71K": "/content/SPair-71k/Layout/small/trn.txt",
  "ALL_VAL_PATH_SPAIR71K": "/content/SPair-71k/Layout/small/val.txt",

  "IMAGE_FOLDER_NAME_SPAIR71K": "/content/SPair-71k/JPEGImages",

  # FOR INFERENCE

  "IMAGE_SIZE": 518,           # 518 for dinov2, 512 for dinov3 and 1024 for sam (TO ASK IF IT'S CORRECT)
	"ALPHA": [0.05, 0.1, 0.2],

  # FOR TUNING

  "TAU": 0.05,
  "LEARNING_RATE": 1e-4,
  "WEIGHT_DECAY": 1e-2,
  "NUM_EPOCHS": 1,
  "NUM_LAYERS": 1,
  "BATCH_SIZE": 16,
  "TUNING": True
}

MODEL = None

# --- RESIZE IMAGE TO STANDARD MODEL DIMENSIONS, CONVERT IT INTO A TENSOR AND NORMALIZE IT ---

PREPROCESS = transforms.Compose([
    transforms.Resize((CONFIGURATION["IMAGE_SIZE"], CONFIGURATION["IMAGE_SIZE"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])