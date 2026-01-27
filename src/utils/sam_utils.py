from PIL import Image
import torch, torch.nn.functional as F, math

from ..dataset.spair71k import CONFIGURATION_DS
from ..utils.common_utils import PREPROCESS

CONFIGURATION_MODEL = {  
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PTH_PATH_SAM": "/content/drive/MyDrive/AML-Semantic-Correspondence/weights/sam_vit_b_01ec64.pth",
    "IMAGE_SIZE": 1024,
}

def get_descriptors(img_path, grad, model):
    img = Image.open(img_path).convert("RGB")
    (w, h) = img.size
    input_tensor = PREPROCESS(CONFIGURATION_MODEL["IMAGE_SIZE"])(img).unsqueeze(0).to(CONFIGURATION_MODEL["DEVICE"])

    with torch.set_grad_enabled(grad):
        x = model.image_encoder(input_tensor).permute(0, 2, 3, 1)
                    
    x = F.normalize(x, dim=-1)
    return (x, w, h)