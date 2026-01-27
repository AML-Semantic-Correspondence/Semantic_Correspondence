from PIL import Image
import torch, torch.nn.functional as F, math

from src.dataset.spair71k import CONFIGURATION_DS
from src.utils.common_utils import PREPROCESS

CONFIGURATION_MODEL = {  
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PTH_PATH_DINOV3": "/content/drive/MyDrive/AML-Semantic-Correspondence/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "IMAGE_SIZE": 512,
}

def get_descriptors(img_path, grad, model):
    img = Image.open(img_path).convert("RGB")
    (w, h) = img.size
    input_tensor = PREPROCESS(CONFIGURATION_MODEL["IMAGE_SIZE"])(img).unsqueeze(0).to(CONFIGURATION_MODEL["DEVICE"])

    with torch.set_grad_enabled(grad):
        x = model.forward_features(input_tensor)["x_norm_patchtokens"]
        (B, N, D) = x.shape
        H = int(math.sqrt(N))
        x = x.reshape(B, H, H, D)                     

    x = F.normalize(x, dim=-1)
    return (x, w, h)