import glob
import os

import torch
from omegaconf import OmegaConf

from . import graph_lib, noise_lib, utils
from .model import SEDD
from .model.ema import ExponentialMovingAverage


def load_model_hf(dir, device):
    """Load a model from a HuggingFace repo id / local `save_pretrained` directory."""
    score_model = SEDD.from_pretrained(dir).to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def _find_checkpoint(root_dir):
    """Prefer the resumable `checkpoints-meta` file, then fall back to `checkpoints/`."""
    meta = os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    if os.path.exists(meta):
        return meta
    candidates = glob.glob(os.path.join(root_dir, "checkpoints", "checkpoint_*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {root_dir!r} "
            f"(looked for checkpoints-meta/checkpoint.pth and checkpoints/checkpoint_*.pth)"
        )
    candidates.sort(key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))
    return candidates[-1]


def load_model_local(root_dir, device):
    """Load an experiment directory created by `train.py` (contains `.hydra/config.yaml`)."""
    cfg_path = os.path.join(root_dir, ".hydra", "config.yaml")
    if os.path.exists(cfg_path):
        cfg = OmegaConf.load(cfg_path)
    else:
        cfg = utils.load_hydra_config_from_run(root_dir)

    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)

    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)
    ckpt_path = _find_checkpoint(root_dir)
    loaded_state = torch.load(ckpt_path, map_location=device)

    score_model.load_state_dict(loaded_state["model"])
    if "ema" in loaded_state:
        ema.load_state_dict(loaded_state["ema"])
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

    score_model.eval()
    return score_model, graph, noise


def load_model(root_dir, device):
    """Load a model either from HuggingFace (repo id) or from a local experiment directory."""
    if os.path.isdir(root_dir):
        return load_model_local(root_dir, device)
    return load_model_hf(root_dir, device)
