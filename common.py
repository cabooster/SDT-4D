"""Shared model, checkpoint, device, and configuration helpers."""

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import yaml

from SDT4D import ExcitationLayerZT, SDT4D, ZTConv


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "y", "1"}:
        return True
    if lowered in {"no", "false", "f", "n", "0"}:
        return False
    raise ValueError(f"Expected a boolean value, got: {value}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(patch_x, patch_t, bayesian):
    return SDT4D(
        img_dim=patch_x,
        img_time=patch_t,
        in_channel=1,
        embedding_dim=64,
        num_heads=8,
        hidden_dim=512,
        window_size=7,
        num_transBlock=1,
        attn_dropout_rate=0.1,
        f_maps=(8, 16, 32, 64),
        input_dropout_rate=0,
        bayesian=bayesian,
    )


def configure_model(
    model,
    ztconv_disable_z=False,
    ztconv_disable_thw=False,
    ztconv_z_scale=1.0,
    ztconv_thw_scale=1.0,
    ztconv_disable_residual=False,
    ztconv_res_scale=1.0,
    decoder_skip_disable=False,
    decoder_skip_scale=1.0,
):
    base = model.module if hasattr(model, "module") else model
    for module in base.modules():
        if isinstance(module, ZTConv):
            module.enable_z_path = not ztconv_disable_z
            module.enable_thw_path = not ztconv_disable_thw
            module.z_scale = ztconv_z_scale
            module.thw_scale = ztconv_thw_scale
            module.enable_residual = not ztconv_disable_residual
            module.res_scale = ztconv_res_scale
        elif isinstance(module, ExcitationLayerZT):
            module.enable_skip = not decoder_skip_disable
            module.skip_scale = decoder_skip_scale


def prepare_device(gpu_ids):
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    requested = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    visible_count = torch.cuda.device_count()
    device_ids = list(range(min(len(requested) or 1, visible_count)))
    return torch.device("cuda:0"), device_ids


def maybe_data_parallel(model, device, device_ids):
    model = model.to(device)
    if device.type == "cuda" and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    return model


def _unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state_dict")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


def load_weights(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _unwrap_state_dict(checkpoint)
    base = model.module if hasattr(model, "module") else model
    incompatible = base.load_state_dict(state_dict, strict=False)
    # Older checkpoints may contain parameters from an unused transformer
    # ModuleList named "layers". The forward path never consumed those layers.
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("layers.")]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint. Missing keys: {incompatible.missing_keys}; "
            f"unexpected keys: {unexpected}"
        )


def save_weights(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    base = model.module if hasattr(model, "module") else model
    torch.save(base.state_dict(), checkpoint_path)


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(vars(config), file, sort_keys=True, allow_unicode=True)
