"""E3 - Where does CLIP 'look'? Window-based crack localisation on DeepCrack.

A simplified WinCLIP-style procedure (no training): each image is covered by
square windows at several scales; every window gets p(crack) from a 2-way
softmax [normal, crack]; pixel scores are the mean over covering windows.
Pixel AUROC / AP against the ground-truth masks are reported per condition.
Absolute values will be modest (windows are coarse, cracks are thin); the
point is the *relative* effect of the wording and the failure maps.
"""
import os

import numpy as np
import pandas as pd

from prelim import common, datasets, prompts as pr, stats


def main():
    ap = common.base_argparser(__doc__)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = common.load_config(args.config, args.set)
    P = common.load_yaml(args.prompts)
    ev, e3 = cfg["eval"], cfg["e3"]
    n = int(ev["n_prompts"])
    out = common.run_dir(cfg)

    conditions = [c for c in e3["conditions"] if c in pr.ALL_CONDITIONS]
    errors, _ = pr.check_prompts(P, n, conditions)
    if errors:
        bad = [c for c in conditions if any(f"condition '{c}'" in e for e in errors)]
        if bad and not args.allow_incomplete:
            raise SystemExit(f"prompts.yaml incomplete for conditions {bad}; see e1 errors.")
        conditions = [c for c in conditions if c not in bad]

    imgs, masks = datasets.load_deepcrack(cfg["data"]["deepcrack_root"], e3["splits"], limit=args.limit)
    print(f"[data] DeepCrack for E3: {len(imgs)} images")

    clip = common.Clip(cfg)
    built = pr.build_conditions(P, conditions, n, int(ev["prompt_seed"]), pr.random_vocab())
    n_emb = clip.class_embedding(pr.normal_sets(P, "basic")["plain"])
    crack_key = e3.get("crack_class", "crack")
    c_embs = {c: clip.class_embedding(built[c][crack_key]) for c in conditions}
    scale = float(ev["logit_scale"])

    rng = np.random.default_rng(int(ev["boot_seed"]))
    max_px = int(e3["max_pixels_per_image"])
    pix_scores = {c: [] for c in conditions}
    pix_labels, keep_heat = [], {}
    vis_idx = set(np.linspace(0, len(imgs) - 1, int(e3["n_vis"])).astype(int).tolist())

    from tqdm.auto import tqdm
    for i, (ip, mp) in enumerate(tqdm(list(zip(imgs, masks)), desc="windows")):
        im = common.load_rgb(ip)
        w, h = im.size
        gt = datasets.load_mask(mp, size=(w, h))
        boxes = common.window_boxes(w, h, e3["window_fracs"], e3["stride_frac"])
        we = clip.encode_windows(im, boxes)
        sel = rng.choice(h * w, size=min(max_px, h * w), replace=False)
        pix_labels.append(gt.reshape(-1)[sel])
        for c in conditions:
            T = np.stack([n_emb, c_embs[c]])
            p = common.softmax(scale * we @ T.T)[:, 1]
            heat = common.window_heatmap(p, boxes, w, h)
            pix_scores[c].append(heat.reshape(-1)[sel])
            if i in vis_idx:
                keep_heat.setdefault(i, {})[c] = heat

    from sklearn.metrics import average_precision_score
    n_img = len(imgs)
    bidx = stats.bootstrap_indices(n_img, int(e3["bootstrap"]), int(ev["boot_seed"]))
    rows, dists = [], {}
    y_all = np.concatenate(pix_labels)
    for c in conditions:
        s_all = np.concatenate(pix_scores[c])
        point = stats.auroc(y_all, s_all)
        dist = np.array([stats.auroc(np.concatenate([pix_labels[j] for j in b]),
                                     np.concatenate([pix_scores[c][j] for j in b])) for b in bidx])
        lo, hi = stats.ci(dist)
        dists[c] = (point, dist)
        per_img = [stats.auroc(pix_labels[j], pix_scores[c][j]) for j in range(n_img)]
        rows.append(dict(condition=c, pixel_auroc=point, ci_lo=lo, ci_hi=hi,
                         pixel_ap=float(average_precision_score(y_all, s_all)),
                         mean_image_auroc=float(np.nanmean(per_img)), n_images=n_img,
                         pos_rate=float(y_all.mean())))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "e3_pixel.csv"), index=False)

    comp = []
    for a, b in [("label_attr", "label"), ("label_attr", "waffle"), ("label_llm", "waffle"),
                 ("term_literal", "label"), ("label_llm", "label_attr")]:
        if a in dists and b in dists:
            d, lo, hi, p = stats.paired_diff(dists[a][0], dists[a][1], dists[b][0], dists[b][1])
            comp.append(dict(A=a, B=b, diff=d, ci_lo=lo, ci_hi=hi, p_boot=p))
    pd.DataFrame(comp).to_csv(os.path.join(out, "e3_pairwise.csv"), index=False)
    common.save_run_info(cfg, args.prompts, out, {"E3": {"n_images": n_img, "conditions": conditions}})
    print(res.round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = sorted(keep_heat)
    ncol = 2 + len(conditions)
    fig, axes = plt.subplots(len(order), ncol, figsize=(2.2 * ncol, 2.6 * len(order)), squeeze=False)
    for r, i in enumerate(order):
        im = common.load_rgb(imgs[i])
        gt = datasets.load_mask(masks[i], size=im.size)
        axes[r, 0].imshow(im)
        axes[r, 0].set_title(os.path.basename(imgs[i])[:18], fontsize=7)
        axes[r, 1].imshow(gt, cmap="gray")
        axes[r, 1].set_title("ground truth", fontsize=7)
        for k, c in enumerate(conditions):
            hm = keep_heat[i][c]
            axes[r, 2 + k].imshow(im)
            axes[r, 2 + k].imshow(hm, cmap="jet", alpha=0.45)  # per-map min-max: shows WHERE it looks
            axes[r, 2 + k].set_title(f"{c}\n[{hm.min():.2f}-{hm.max():.2f}]", fontsize=7)
        for a in axes[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "e3_heatmaps.png"), dpi=120)
    plt.close(fig)
    print(f"[done] E3 results in {out}")


if __name__ == "__main__":
    main()
