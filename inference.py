import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import timm


@dataclass
class InferenceConfig:
    backbone_name: str = "tf_efficientnet_b0"
    img_size: int = 224
    device: str = "cpu"


class DRModel(nn.Module):
    """
    Matches checkpoint keys like:
      backbone.blocks...
      backbone.conv_head...
      fc.weight / fc.bias

    This is the common pattern:
      self.backbone = timm.create_model(..., num_classes=0, global_pool="")
      self.pool = AdaptiveAvgPool2d(1)
      self.fc = Linear(feature_dim, 1)
    """
    def __init__(self, backbone_name: str = "tf_efficientnet_b0", pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )

        # EfficientNet-B0 feature dim is typically 1280.
        # If your training used a different dim, we'll auto-fix at load time if needed.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)   # [B, C, H, W]
        feat = self.pool(feat).flatten(1)          # [B, C]
        logit = self.fc(feat).squeeze(1)           # [B]
        return logit


def load_temperature(calib_json_path: str) -> float:
    if not os.path.exists(calib_json_path):
        return 1.0
    with open(calib_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("temperature", 1.0))


def preprocess_image(pil_img: Image.Image, img_size: int = 224) -> torch.Tensor:
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(img).float()


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        return ckpt
    return ckpt


def load_model(model_path: str, cfg: InferenceConfig) -> nn.Module:
    model = DRModel(backbone_name=cfg.backbone_name, pretrained=False)

    ckpt = torch.load(model_path, map_location=cfg.device)
    state = _extract_state_dict(ckpt)

    if not isinstance(state, dict) or len(state) == 0:
        raise RuntimeError("Could not extract a valid state_dict from the checkpoint.")

    # Remove DataParallel prefix if it exists
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # If your checkpoint keys start with "backbone." and "fc.", this will match.
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Auto-fix for feature dimension mismatch (rare, but can happen)
    # If fc.weight is unexpected/missing due to different in_features, rebuild fc.
    # We detect actual feature dim by running a dummy forward_features.
    if any(k.startswith("fc.") for k in missing) or any(k.startswith("fc.") for k in unexpected):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.img_size, cfg.img_size, device=cfg.device)
            feat = model.backbone.forward_features(dummy)
            feat = model.pool(feat).flatten(1)
            in_dim = feat.shape[1]
        model.fc = nn.Linear(in_dim, 1).to(cfg.device)
        missing, unexpected = model.load_state_dict(state, strict=False)

    if len(missing) > 50:
        raise RuntimeError(
            f"Too many missing keys ({len(missing)}). "
            "Checkpoint likely doesn't match tf_efficientnet_b0 backbone+fc structure."
        )

    model.to(cfg.device)
    model.eval()
    return model


@torch.no_grad()
def predict_single(model: nn.Module, pil_img: Image.Image, cfg: InferenceConfig, temperature: float = 1.0) -> dict:
    x = preprocess_image(pil_img, img_size=cfg.img_size).unsqueeze(0).to(cfg.device)
    logit = float(model(x)[0].item())
    cal_logit = logit / max(float(temperature), 1e-6)
    prob = _sigmoid(cal_logit)

    return {
        "raw_logit": logit,
        "temperature": float(temperature),
        "calibrated_probability": prob
    }