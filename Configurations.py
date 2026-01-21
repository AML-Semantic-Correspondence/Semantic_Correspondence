# @title Configurations

from torchvision import transforms
import torch

# --- CONFIGURATIONS: SET PATHS, DEVICE, MODEL AND HYPERPARAMETERS ---

CONFIGURATION = {
	"PATH_DRIVE": "/content/drive",
	"PATH_FILE": "/content/drive/MyDrive/AML-Semantic-Correspondence/datasets/SPair-71k.tar.gz",
  "PATH_BEST_MODEL": "/content/drive/MyDrive/AML-Semantic-Correspondence/weights/best_model.pth",

	"PTH_PATH_DINOV2": "/content/drive/MyDrive/AML-Semantic-Correspondence/weights/dinov2_vitb14_reg4_pretrain.pth",
  "PTH_PATH_DINOV3": "/content/drive/MyDrive/AML-Semantic-Correspondence/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
  "PTH_PATH_SAM": "/content/drive/MyDrive/AML-Semantic-Correspondence/weights/sam_vit_b_01ec64.pth",

  "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
  "MODEL_VERSION": "dinov2",                                        # "dinov2", "dinov3", "sam"

  # DATASET SPAIR-71k

  "PATH_TEST_SPAIR71K": "/content/SPair-71k/PairAnnotation/test",
  "PATH_VAL_SPAIR71K": "/content/SPair-71k/PairAnnotation/val",
  "PATH_TRAIN_SPAIR71K": "/content/SPair-71k/PairAnnotation/trn",

  "ALL_TEST_PATH_SPAIR71K": "/content/SPair-71k/Layout/large/test.txt",
  "ALL_TRAIN_PATH_SPAIR71K": "/content/SPair-71k/Layout/large/trn.txt",
  "ALL_VAL_PATH_SPAIR71K": "/content/SPair-71k/Layout/large/val.txt",

  "IMAGE_FOLDER_NAME_SPAIR71K": "/content/SPair-71k/JPEGImages",

  # FOR INFERENCE

	"ALPHA": [0.05, 0.1, 0.2],
  "USE_WIN": False,
  "WINDOW_SOFTMAX": 7,
  "TAU_SOFTMAX": 0.07,

  # FOR TUNING

  "TAU": 0.05,
  "LEARNING_RATE": 5e-5,
  "WEIGHT_DECAY": 5e-2,
  "NUM_EPOCHS": 1,
  "NUM_LAYERS": 1,
  "BATCH_SIZE": 32,
  "TUNING": True
}

MODEL = None

if CONFIGURATION["MODEL_VERSION"] == "dinov2":              # DINOV2 -> DIM PATCH -> 14 -> IMAGE SIZE 518
  IMAGE_SIZE = 518
elif CONFIGURATION["MODEL_VERSION"] == "dinov3":            # DINOV3 -> DIM PATCH -> 16 -> IMAGE SIZE 512
  IMAGE_SIZE = 512
else:
  IMAGE_SIZE = 1024                                         # SAM -> DIM PATCH -> 16 -> IMAGE SIZE 1024

# --- RESIZE IMAGE TO STANDARD MODEL DIMENSIONS, CONVERT IT INTO A TENSOR AND NORMALIZE IT ---

PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])