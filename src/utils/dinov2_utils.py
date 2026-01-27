from PIL import Image
import torch, torch.nn.functional as F, math

from src.dataset.spair71k import CONFIGURATION_DS
from src.utils.common_utils import PREPROCESS

CONFIGURATION_MODEL = {  
    "PTH_PATH_DINOV2": "/content/drive/MyDrive/AML-Semantic-Correspondence/weights/dinov2_vitb14_reg4_pretrain.pth",
    "IMAGE_SIZE": 518,
}

def get_descriptors(img_path, grad, model):
    img = Image.open(img_path).convert("RGB")
    (w, h) = img.size
    input_tensor = PREPROCESS(CONFIGURATION_MODEL["IMAGE_SIZE"])(img).unsqueeze(0).to(CONFIGURATION_MODEL["DEVICE"])

    with torch.set_grad_enabled(grad):
        x = model.get_intermediate_layers(input_tensor, n=1)[0]   
        (B, N, D) = x.shape
        H = int(math.sqrt(N))
        x = x.reshape(B, H, H, D)                

    x = F.normalize(x, dim=-1)
    return (x, w, h)