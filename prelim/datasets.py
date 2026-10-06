"""Dataset readers.

CODEBRIM classification dataset (Mundt et al., CVPR 2019), layout after unzip:
    <root>/metadata/background.xml, <root>/metadata/defects.xml
    <root>/{train,val,test}/{background,defects}/*.png
Each XML child element carries the crop file name as its (only) attribute and
six 0/1 children: Background, Crack, Spallation, Efflorescence, ExposedBars, CorrosionStain.

DeepCrack (Liu et al., Neurocomputing 2019), layout after unzip:
    <root>/{train,test}_img/*.jpg and <root>/{train,test}_lab/*.png  (binary masks)
"""
import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

CODEBRIM_TAGS = ["Background", "Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain"]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_codebrim_root(path):
    """Accept either the dataset root itself or any parent folder of it."""
    if os.path.exists(os.path.join(path, "metadata", "defects.xml")):
        return path
    hits = glob.glob(os.path.join(path, "**", "metadata", "defects.xml"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"metadata/defects.xml not found under {path}. See README step 3.")
    hits.sort(key=len)
    return os.path.dirname(os.path.dirname(hits[0]))


def _parse_xml(xml_path):
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    root = ET.parse(xml_path).getroot()
    out = {}
    for el in root:
        if not el.attrib:
            continue
        name = list(el.attrib.values())[0]
        children = list(el)
        by_tag = {ch.tag: ch.text for ch in children}
        if all(t in by_tag for t in CODEBRIM_TAGS):
            vec = [int(str(by_tag[t]).strip() == "1") for t in CODEBRIM_TAGS]
        else:  # fall back to positional order, as in the official loader
            vec = [int(str(ch.text).strip() == "1") for ch in children[:6]]
        out[(stem, name)] = vec
        out.setdefault(("*", name), vec)
    return out


def load_codebrim(root, split, limit=None, seed=0):
    root = find_codebrim_root(root)
    labels = {}
    for x in ("background.xml", "defects.xml"):
        labels.update(_parse_xml(os.path.join(root, "metadata", x)))
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"split folder not found: {split_dir}")
    paths, ys, missing = [], [], 0
    for sub in sorted(os.listdir(split_dir)):
        d = os.path.join(split_dir, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(IMG_EXT):
                continue
            ext = os.path.splitext(fn)[1]
            key_name = fn.split("_-_")[0] + ext if "_-_" in fn else fn  # replicated copies
            vec = labels.get((sub, key_name)) or labels.get(("*", key_name))
            if vec is None:
                missing += 1
                continue
            paths.append(os.path.join(d, fn))
            ys.append(vec)
    if missing:
        print(f"[warn] {missing} images in {split_dir} had no label entry and were skipped")
    if not paths:
        raise RuntimeError(f"no labelled images found in {split_dir}")
    y = np.array(ys, dtype=np.int64)
    if limit and limit < len(paths):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(paths), size=limit, replace=False))
        paths = [paths[i] for i in keep]
        y = y[keep]
    return paths, y


def find_deepcrack_root(path):
    if os.path.isdir(os.path.join(path, "test_img")):
        return path
    hits = glob.glob(os.path.join(path, "**", "test_img"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"test_img/ not found under {path}. See README step 3.")
    hits.sort(key=len)
    return os.path.dirname(hits[0])


def load_deepcrack(root, splits=("train", "test"), limit=None, seed=0):
    root = find_deepcrack_root(root)
    imgs, masks = [], []
    for sp in splits:
        for p in sorted(glob.glob(os.path.join(root, f"{sp}_img", "*"))):
            if not p.lower().endswith(IMG_EXT):
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            cand = glob.glob(os.path.join(root, f"{sp}_lab", stem + ".*"))
            if cand:
                imgs.append(p)
                masks.append(cand[0])
    if not imgs:
        raise RuntimeError(f"no DeepCrack image/mask pairs under {root}")
    if limit and limit < len(imgs):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(imgs), size=limit, replace=False))
        imgs = [imgs[i] for i in keep]
        masks = [masks[i] for i in keep]
    return imgs, masks


def load_mask(path, size=None):
    m = Image.open(path).convert("L")
    if size is not None and m.size != size:
        m = m.resize(size, Image.NEAREST)
    return np.array(m) > 127


def severity_proxies(mask):
    """Scale-free and pixel-scale proxies of crack severity from a binary mask.

    area_ratio : crack pixels / image pixels
    width_p90  : 90th percentile of local crack width along the skeleton (pixels)
    length     : skeleton length / image diagonal
    NOTE: width is in pixels; without a physical scale it is NOT the mm width
    used by the inspection manual. It is only a relative proxy.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    h, w = mask.shape
    area = mask.mean()
    if mask.sum() == 0:
        return {"area_ratio": 0.0, "width_p90": 0.0, "length": 0.0}
    dt = distance_transform_edt(mask)
    skel = skeletonize(mask)
    widths = 2.0 * dt[skel] if skel.any() else np.array([0.0])
    return {
        "area_ratio": float(area),
        "width_p90": float(np.percentile(widths, 90)),
        "length": float(skel.sum() / np.hypot(h, w)),
    }
