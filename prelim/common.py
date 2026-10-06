"""Shared utilities for the preliminary experiments.

Everything that touches the CLIP model lives here, so every experiment
uses exactly the same preprocessing, precision and normalisation.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
import platform
import sys

import numpy as np
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # CODEBRIM crops can be very large

CLIP_MEAN_RGB = (122, 116, 104)  # CLIP normalisation mean * 255, used as padding colour


# --------------------------------------------------------------------------- config
def base_argparser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--prompts", default="prompts.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config values, e.g. --set model.arch=ViT-B-16 run_name=vitb16")
    p.add_argument("--allow_incomplete", action="store_true",
                   help="run even if prompts.yaml still contains TODO items (smoke tests only)")
    return p


def load_config(path, overrides=()):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"bad override '{ov}', expected KEY=VALUE")
        key, val = ov.split("=", 1)
        val = yaml.safe_load(val)
        d = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            d = d.setdefault(k, {})
        d[parts[-1]] = val
    return cfg


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dir(cfg):
    path = os.path.join(cfg["paths"]["results_dir"], cfg["run_name"])
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "figures"), exist_ok=True)
    return path


def cache_dir(cfg):
    path = os.path.join(cfg["paths"]["results_dir"], "_cache")
    os.makedirs(path, exist_ok=True)
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_run_info(cfg, prompts_path, out_dir, extra=None):
    """Record everything needed to reproduce a run (appended per experiment)."""
    import torch
    try:
        import open_clip
        oc_ver = open_clip.__version__
    except Exception:  # pragma: no cover
        oc_ver = "unknown"
    info_path = os.path.join(out_dir, "run_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
    info.update({
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "open_clip": oc_ver,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": cfg,
        "prompts_sha256": sha256_file(prompts_path),
    })
    if extra:
        info.setdefault("experiments", {}).update(extra)
    write_json(info, info_path)


# --------------------------------------------------------------------------- images
def letterbox(img, fill=CLIP_MEAN_RGB):
    """Pad to a square without cropping, so no part of a defect is cut off."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def load_rgb(path):
    return Image.open(path).convert("RGB")


def window_boxes(w, h, fracs, stride_frac):
    """Square sliding windows (x0, y0, x1, y1) at several scales, covering the borders."""
    boxes = []
    short = min(w, h)
    for frac in fracs:
        side = max(16, int(round(short * frac)))
        stride = max(1, int(round(side * stride_frac)))
        xs = list(range(0, max(w - side, 0) + 1, stride))
        ys = list(range(0, max(h - side, 0) + 1, stride))
        if xs[-1] != max(w - side, 0):
            xs.append(max(w - side, 0))
        if ys[-1] != max(h - side, 0):
            ys.append(max(h - side, 0))
        for y in ys:
            for x in xs:
                boxes.append((x, y, min(x + side, w), min(y + side, h)))
    return boxes


def prepare_image(img, preprocess, mode):
    img.thumbnail((1024, 1024))  # final input is 224 px anyway; avoids huge canvases
    if mode == "letterbox":
        img = letterbox(img)
    return preprocess(img)


class _ImageDataset:
    """Module-level (picklable) dataset so DataLoader workers also work on Windows."""

    def __init__(self, paths, preprocess, mode):
        self.paths, self.preprocess, self.mode = paths, preprocess, mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return prepare_image(load_rgb(self.paths[i]), self.preprocess, self.mode)


# --------------------------------------------------------------------------- model
class Clip:
    """Thin wrapper around an OpenCLIP model.

    All returned embeddings are L2-normalised float32 numpy arrays.
    """

    def __init__(self, cfg):
        import torch
        import open_clip

        self.torch = torch
        mcfg = cfg["model"]
        dev = mcfg.get("device", "auto")
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev
        pretrained = mcfg.get("pretrained") or None
        self.arch = mcfg["arch"]
        self.pretrained = pretrained or "random-init"
        model, _, preprocess = open_clip.create_model_and_transforms(self.arch, pretrained=pretrained)
        self.model = model.eval().to(dev)
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.arch)
        self.use_amp = dev == "cuda" and mcfg.get("precision", "fp16") == "fp16"
        self.batch_size = int(cfg.get("runtime", {}).get("batch_size", 64))
        self.num_workers = int(cfg.get("runtime", {}).get("num_workers", 2))
        self.image_mode = cfg.get("image", {}).get("preprocess", "letterbox")

    def _amp(self):
        if self.use_amp:
            return self.torch.autocast(device_type="cuda", dtype=self.torch.float16)
        return contextlib.nullcontext()

    @staticmethod
    def _norm(x):
        x = x.float()
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def encode_text(self, texts):
        torch = self.torch
        out = []
        with torch.no_grad(), self._amp():
            for i in range(0, len(texts), 256):
                tok = self.tokenizer(texts[i:i + 256]).to(self.device)
                out.append(self._norm(self.model.encode_text(tok)).cpu())
        return torch.cat(out).numpy()

    def class_embedding(self, texts):
        """Prompt ensembling as in the CLIP paper: mean of normalised embeddings, renormalised."""
        e = self.encode_text(list(texts)).mean(0)
        return e / max(np.linalg.norm(e), 1e-12)

    def encode_tensor_batch(self, x):
        torch = self.torch
        with torch.no_grad(), self._amp():
            return self._norm(self.model.encode_image(x.to(self.device))).cpu().numpy()

    def prepare(self, img):
        return prepare_image(img, self.preprocess, self.image_mode)

    def encode_paths(self, paths, cache_path=None, desc="images"):
        """Encode whole images (with caching, since text-side changes never need re-encoding)."""
        torch = self.torch
        from tqdm.auto import tqdm

        key = hashlib.sha256(("\n".join(paths) + f"|{self.arch}|{self.pretrained}|{self.image_mode}")
                             .encode()).hexdigest()[:16]
        if cache_path:
            cache_path = f"{cache_path}_{key}.npz"
            if os.path.exists(cache_path) and self.pretrained != "random-init":
                data = np.load(cache_path)
                print(f"[cache] loaded {desc} embeddings from {cache_path}")
                return data["emb"].astype(np.float32)

        dl = torch.utils.data.DataLoader(_ImageDataset(paths, self.preprocess, self.image_mode),
                                         batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.num_workers)
        embs = [self.encode_tensor_batch(x) for x in tqdm(dl, desc=f"encode {desc}")]
        emb = np.concatenate(embs).astype(np.float32)
        if cache_path and self.pretrained != "random-init":
            np.savez_compressed(cache_path, emb=emb.astype(np.float16))
        return emb

    def encode_windows(self, img, boxes):
        """Encode square crops of one image; returns (n_boxes, d)."""
        torch = self.torch
        crops = [self.preprocess(img.crop(b)) for b in boxes]
        out = []
        for i in range(0, len(crops), self.batch_size):
            out.append(self.encode_tensor_batch(torch.stack(crops[i:i + self.batch_size])))
        return np.concatenate(out)


def softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def window_heatmap(scores, boxes, w, h):
    """Average window scores over every pixel each window covers."""
    acc = np.zeros((h, w), np.float64)
    cnt = np.zeros((h, w), np.float64)
    for s, (x0, y0, x1, y1) in zip(scores, boxes):
        acc[y0:y1, x0:x1] += s
        cnt[y0:y1, x0:x1] += 1
    return (acc / np.maximum(cnt, 1)).astype(np.float32)
