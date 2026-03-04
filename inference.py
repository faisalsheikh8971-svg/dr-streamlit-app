# inference.py
import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import timm


# ----------------------------
# Config
# ----------------------------
@dataclass
class InferenceConfig:
    backbone_name: str = "tf_efficientnet_b0"
    img_size: int = 224
    device: str = "cpu"


# ----------------------------
# Model
# ----------------------------
class DRModel(nn.Module):
    """
    timm backbone with num_classes=1 -> outputs 1 logit
    """
    def __init__(self, backbone_name: str = "tf_efficientnet_b0", pretrained: bool = False):
        super().__init__()
        self.net = timm.create_model(backbone_name, pretrained=pretrained, num_classes=1)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ----------------------------
# Utilities
# ----------------------------
def load_temperature(calib_json_path: str) -> float:
    if not os.path.exists(calib_json_path):
        return 1.0
    with open(calib_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("temperature", 1.0))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def preprocess_image(pil_img: Image.Image, img_size: int = 224) -> torch.Tensor:
    """
    Streamlit-cloud-safe preprocessing (NO cv2)
    - RGB
    - resize with PIL
    - [0,1] float32
    - CHW
    """
    img = pil_img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr).float()


def load_model(model_path: str, cfg: InferenceConfig) -> nn.Module:
    model = DRModel(backbone_name=cfg.backbone_name, pretrained=False)

    ckpt = torch.load(model_path, map_location=cfg.device)

    # extract state dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        raise RuntimeError("Checkpoint format not supported.")

    # strip common prefixes
    def strip_prefix(sd, prefix):
        out = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
            else:
                out[k] = v
        return out

    state = strip_prefix(state, "module.")
    state = strip_prefix(state, "model.")
    state = strip_prefix(state, "backbone.")

    # your wrapper uses self.net -> keys should be net.*
    any_key = next(iter(state.keys()))
    if not any_key.startswith("net."):
        state = {("net." + k): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    # sanity check
    if len(missing) > 50:
        raise RuntimeError(
            "Too many missing keys while loading weights. Check that backbone_name matches training."
        )

    model.to(cfg.device)
    model.eval()
    return model


@torch.no_grad()
def predict_single(model: nn.Module, pil_img: Image.Image, cfg: InferenceConfig, temperature: float = 1.0) -> dict:
    x = preprocess_image(pil_img, img_size=cfg.img_size).unsqueeze(0).to(cfg.device)
    logit = float(model(x)[0].item())
    cal_logit = logit / max(temperature, 1e-6)
    prob = _sigmoid(cal_logit)

    return {
        "raw_logit": logit,
        "temperature": float(temperature),
        "calibrated_probability": prob,
    }
