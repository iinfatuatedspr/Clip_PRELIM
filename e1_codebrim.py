"""E1 - Does the wording of the damage description change zero-shot recognition?

Targets
  T1  anomaly     : background (normal) vs any defect, score = 1 - p(normal)
  T2  per class   : one-vs-rest AUROC for crack / spallation / efflorescence /
                    exposed bars / corrosion stain, score = p(class)
Scoring is standard CLIP zero-shot classification: softmax over
[normal subclasses..., 5 defect classes] with logit scale 100.
The normal prompts are identical across conditions, so only the defect
wording varies between conditions.
"""
import os

import numpy as np
import pandas as pd

from prelim import common, datasets, prompts as pr, stats

COMPARISONS = [  # (A, B, question answered by A - B)
    ("term_literal", "label", "直译点検要领用语 vs 通用英语标签"),
    ("attr", "label", "只用视觉属性描述(无类别名) vs 标签"),
    ("label_attr", "label", "标签+视觉属性 vs 标签"),
    ("label_attr", "waffle", "视觉属性的内容 vs 同数量随机词(集成效应对照)"),
    ("label_llm", "label_attr", "LLM言い換え扩展 vs 人手属性描述"),
    ("label_llm", "waffle", "LLM言い換え vs 随机词对照"),
    ("llm", "attr", "LLM言い換え vs 人手属性描述(均无类别名)"),
]


def main():
    ap = common.base_argparser(__doc__)
    ap.add_argument("--limit", type=int, default=None, help="use a random subset (smoke test only)")
    ap.add_argument("--split", default=None, help="override data.codebrim_split (e.g. val for smoke runs)")
    ap.add_argument("--no_vis", action="store_true")
    args = ap.parse_args()
    cfg = common.load_config(args.config, args.set)
    P = common.load_yaml(args.prompts)
    ev = cfg["eval"]
    n = int(ev["n_prompts"])
    classes = list(P["classes"].keys())

    conditions = [c for c in ev["conditions"] if c in pr.ALL_CONDITIONS]
    errors, warns = pr.check_prompts(P, n, conditions)
    for w in warns:
        print("[warn]", w)
    if errors:
        for e in errors:
            print("[error]", e)
        if not args.allow_incomplete:
            raise SystemExit("prompts.yaml is incomplete (see errors above). Fix it, or use --allow_incomplete "
                             "for a smoke run only.")
        # drop conditions that cannot be built
        conditions = [c for c in conditions
                      if not any(f"condition '{c}'" in e for e in errors)]
        print("[info] running with conditions:", conditions)

    out = common.run_dir(cfg)
    split = args.split or cfg["data"]["codebrim_split"]
    paths, Y = datasets.load_codebrim(cfg["data"]["codebrim_root"], split, limit=args.limit)
    tags = datasets.CODEBRIM_TAGS
    col = {t: i for i, t in enumerate(tags)}
    is_bg = Y[:, col["Background"]].astype(bool)
    print(f"[data] CODEBRIM/{split}: {len(paths)} images, {is_bg.sum()} background, "
          + ", ".join(f"{t}={Y[:, i].sum()}" for i, t in enumerate(tags[1:], 1)))

    clip = common.Clip(cfg)
    img = clip.encode_paths(paths, os.path.join(common.cache_dir(cfg),
                                                f"codebrim_{split}_{clip.arch}_{clip.pretrained}"), "CODEBRIM")

    vocab = pr.random_vocab()
    built = pr.build_conditions(P, conditions, n, int(ev["prompt_seed"]), vocab)
    used = {"conditions": built,
            "normal": {k: pr.normal_sets(P, k) for k in ev["normal_sets"]}}
    common.write_json(used, os.path.join(out, "e1_prompts_used.json"))

    idx = stats.bootstrap_indices(len(paths), int(ev["bootstrap"]), int(ev["boot_seed"]))
    scale = float(ev["logit_scale"])
    rows, dists, score_frames, confusions = [], {}, [], []

    for nset in ev["normal_sets"]:
        normal = pr.normal_sets(P, nset)
        n_emb = np.stack([clip.class_embedding(v) for v in normal.values()])
        k = n_emb.shape[0]
        for cond in conditions:
            c_emb = np.stack([clip.class_embedding(built[cond][c]) for c in classes])
            probs = common.softmax(scale * img @ np.concatenate([n_emb, c_emb]).T)
            p_normal = probs[:, :k].sum(1)
            p_cls = probs[:, k:]

            targets = {"T1_anomaly": (~is_bg, 1.0 - p_normal)}
            for j, c in enumerate(classes):
                tag = P["classes"][c]["codebrim_tag"]
                targets[c] = (Y[:, col[tag]].astype(bool), p_cls[:, j])
            macro_dist = []
            for tname, (y, s) in targets.items():
                point, dist = stats.boot_auroc(y, s, idx)
                lo, hi = stats.ci(dist)
                dists[(nset, cond, tname)] = (point, dist)
                rows.append(dict(normal_set=nset, condition=cond, target=tname, auroc=point,
                                 ci_lo=lo, ci_hi=hi, n_pos=int(y.sum()), n_neg=int((~y).sum())))
                if tname != "T1_anomaly":
                    macro_dist.append(dist)
            macro_point = float(np.nanmean([dists[(nset, cond, c)][0] for c in classes]))
            md = np.nanmean(np.stack(macro_dist), axis=0)
            lo, hi = stats.ci(md)
            dists[(nset, cond, "macro_T2")] = (macro_point, md)
            rows.append(dict(normal_set=nset, condition=cond, target="macro_T2", auroc=macro_point,
                             ci_lo=lo, ci_hi=hi, n_pos=-1, n_neg=-1))

            sf = pd.DataFrame({"path": paths, "normal_set": nset, "condition": cond,
                               "p_normal": p_normal})
            for j, c in enumerate(classes):
                sf[f"p_{c}"] = p_cls[:, j]
            for t in tags:
                sf[f"y_{t}"] = Y[:, col[t]]
            score_frames.append(sf)

            # confusion among single-label defect images: which damage words get mixed up?
            single = (~is_bg) & (Y[:, 1:].sum(1) == 1)
            if single.any():
                tag2cls = {P["classes"][c]["codebrim_tag"]: c for c in classes}
                true_idx = [tag2cls[tags[1 + int(np.argmax(r))]] for r in Y[single, 1:]]
                pred_idx = [classes[int(np.argmax(r))] for r in p_cls[single]]
                cm = pd.crosstab(pd.Series(true_idx, name="true"), pd.Series(pred_idx, name="pred"))
                cm = cm.reindex(index=classes, columns=classes, fill_value=0)
                cm.insert(0, "condition", cond)
                cm.insert(0, "normal_set", nset)
                confusions.append(cm.reset_index())

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "e1_auroc.csv"), index=False)

    comp_rows = []
    for nset in ev["normal_sets"]:
        for a, b, q in COMPARISONS:
            if a not in conditions or b not in conditions:
                continue
            for tname in ["T1_anomaly", "macro_T2"] + classes:
                pa, da = dists[(nset, a, tname)]
                pb, db = dists[(nset, b, tname)]
                d, lo, hi, p = stats.paired_diff(pa, da, pb, db)
                comp_rows.append(dict(normal_set=nset, A=a, B=b, question=q, target=tname,
                                      diff=d, ci_lo=lo, ci_hi=hi, p_boot=p))
    pd.DataFrame(comp_rows).to_csv(os.path.join(out, "e1_pairwise.csv"), index=False)

    # effect of hard-negative normal prompts on T1 (same defect wording)
    if len(ev["normal_sets"]) > 1 and "basic" in ev["normal_sets"] and "extended" in ev["normal_sets"]:
        nrows = []
        for cond in conditions:
            pa, da = dists[("extended", cond, "T1_anomaly")]
            pb, db = dists[("basic", cond, "T1_anomaly")]
            d, lo, hi, p = stats.paired_diff(pa, da, pb, db)
            nrows.append(dict(condition=cond, auroc_basic=pb, auroc_extended=pa, diff=d, ci_lo=lo, ci_hi=hi,
                              p_boot=p))
        pd.DataFrame(nrows).to_csv(os.path.join(out, "e1_normal_sets.csv"), index=False)

    pd.concat(score_frames).to_csv(os.path.join(out, "e1_scores.csv.gz"), index=False)
    if confusions:
        pd.concat(confusions).to_csv(os.path.join(out, "e1_confusion.csv"), index=False)

    common.save_run_info(cfg, args.prompts, out, {"E1": {"split": split, "n_images": len(paths),
                                                          "conditions": conditions}})
    print(res[res.normal_set == ev["normal_sets"][0]].pivot(index="condition", columns="target",
                                                            values="auroc").round(3).to_string())

    if not args.no_vis:
        visualise_failures(cfg, P, clip, paths, Y, dists, built, out, classes)
    print(f"[done] E1 results in {out}")


def visualise_failures(cfg, P, clip, paths, Y, dists, built, out, classes):
    """Top false positives (background scored as damage) and false negatives, with window heatmaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = cfg["eval"]
    cond = ev.get("vis_condition", "label_attr")
    if cond not in built:
        cond = list(built.keys())[0]
    scores = pd.read_csv(os.path.join(out, "e1_scores.csv.gz"))
    s = scores[(scores.condition == cond) & (scores.normal_set == ev["normal_sets"][0])].reset_index(drop=True)
    s["anomaly"] = 1.0 - s["p_normal"]
    k = int(ev.get("n_vis", 6))
    fp = s[s.y_Background == 1].sort_values("anomaly", ascending=False).head(k)
    fn = s[s.y_Background == 0].sort_values("anomaly", ascending=True).head(k)

    normal = pr.normal_sets(P, ev["normal_sets"][0])
    n_emb = np.stack([clip.class_embedding(v) for v in normal.values()])
    c_emb = np.stack([clip.class_embedding(built[cond][c]) for c in classes])
    T = np.concatenate([n_emb, c_emb]).T
    kk = n_emb.shape[0]
    e3 = cfg["e3"]

    for name, df in (("false_positives", fp), ("false_negatives", fn)):
        if df.empty:
            continue
        fig, axes = plt.subplots(2, len(df), figsize=(3.2 * len(df), 6.6), squeeze=False)
        for j, (_, r) in enumerate(df.iterrows()):
            im = common.load_rgb(r["path"])
            im.thumbnail((768, 768))
            w, h = im.size
            boxes = common.window_boxes(w, h, e3["window_fracs"], e3["stride_frac"])
            we = clip.encode_windows(im, boxes)
            pw = common.softmax(float(ev["logit_scale"]) * we @ T)
            heat = common.window_heatmap(1.0 - pw[:, :kk].sum(1), boxes, w, h)
            true = [t for t in datasets.CODEBRIM_TAGS if r[f"y_{t}"] == 1]
            top = classes[int(np.argmax([r[f"p_{c}"] for c in classes]))]
            axes[0, j].imshow(im)
            axes[0, j].set_title(f"true: {','.join(true)}\nanomaly={r['anomaly']:.2f} top={top}", fontsize=8)
            axes[1, j].imshow(im)
            axes[1, j].imshow(heat, cmap="jet", alpha=0.45)  # per-map min-max
            axes[1, j].set_title(f"{os.path.basename(r['path'])[:24]}\nheat [{heat.min():.2f}-{heat.max():.2f}]",
                                 fontsize=7)
            for a in axes[:, j]:
                a.axis("off")
        fig.suptitle(f"E1 {name} (condition={cond}); bottom row: window anomaly heatmap", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "figures", f"e1_{name}.png"), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
