# @title Main

# !pip install torchmetrics                # ONLY FOR FIRST EXECUTION

from google.colab import drive
from Inference import run_evaluation
from Configurations import CONFIGURATION, Loader
import torch, time

# --- IF SAM MODEL ---

if CONFIGURATION["MODEL_VERSION"] == "sam":
    # !pip install git+https://github.com/facebookresearch/segment-anything.git
    from segment_anything import sam_model_registry                 # DOWNLOAD

# --- MOUNT DRIVE AND EXTRACT DATASET ---

drive.mount(CONFIGURATION["PATH_DRIVE"], force_remount=True)
# !tar -xzf {CONFIGURATION["PATH_FILE"]}

# --- LOAD MODEL ---

print()
print("Loading ", CONFIGURATION["MODEL_VERSION"], " model...")

if CONFIGURATION["MODEL_VERSION"] == "dinov2":
    MODEL = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg", weights=CONFIGURATION["PTH_PATH"])
elif CONFIGURATION["MODEL_VERSION"] == "dinov3":
    MODEL = torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16", weights=CONFIGURATION["PTH_PATH"])
elif CONFIGURATION["MODEL_VERSION"] == "sam":
    MODEL = sam_model_registry["vit_b"](checkpoint=CONFIGURATION["PTH_PATH"])
MODEL = MODEL.to(CONFIGURATION["DEVICE"])

if CONFIGURATION["DEVICE"] == "cuda":
    torch.cuda.synchronize()                         # SYNCHROIZE GPU

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)          # GO
    start_event.record()

else:
    start_event = time.time()       # GO

loader = Loader(0, batch_size=1)
run_evaluation(loader, "test")

if CONFIGURATION["DEVICE"] == "cuda":
    torch.cuda.synchronize()          # STOP
    end_event.record()
else:
    end_event = time.time()


elapsed_time = start_event.elapsed_time(end_event) / 1000              # SECONDS
print()
print("Analysis for: " + CONFIGURATION["DEVICE"])
print("Total required time: " + str(elapsed_time) + " seconds")