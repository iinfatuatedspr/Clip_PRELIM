"""E2 - Do degree expressions (slight -> moderate -> severe) keep their order inside CLIP?

Two checks, per degree-wording set in prompts.yaml:
  text side : is the 'moderate' embedding between 'slight' and 'severe'?
              t = projection of (moderate - slight) onto (severe - slight); 0 < t < 1 means ordered.
  image side: expected level E = sum_k k * p(level k) (softmax over the 3 levels),
              Spearman correlation with mask-based severity proxies on DeepCrack.
A random-word control set ("a <random word> crack" in arbitrary order) is added
automatically to show what 'no ordinal information' looks like.
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
    ev = cfg["eval"]
    out = common.run_dir(cfg)

    imgs, masks = datasets.load_deepcrack(cfg["data"]["deepcrack_root"], cfg["data"]["deepcrack_splits"],
                                          limit=args.limit)
    print(f"[data] DeepCrack: {len(imgs)} images")

    import hashlib
    key = hashlib.sha256("\n".join(imgs).encode()).hexdigest()[:12]
    proxy_path = os.path.join(common.cache_dir(cfg), f"deepcrack_proxies_{key}.csv")
    if os.path.exists(proxy_path) and args.limit is None:
        prox = pd.read_csv(proxy_path)
    else:
        from tqdm.auto import tqdm
        prox = pd.DataFrame([{"path": p, **datasets.severity_proxies(datasets.load_mask(m))}
                             for p, m in tqdm(list(zip(imgs, masks)), desc="mask proxies")])
        if args.limit is None:
            prox.to_csv(proxy_path, index=False)
    proxies = ["area_ratio", "width_p90", "length"]

    clip = common.Clip(cfg)
    img = clip.encode_paths(imgs, os.path.join(common.cache_dir(cfg),
                                               f"deepcrack_{clip.arch}_{clip.pretrained}"), "DeepCrack")
    vocab = pr.random_vocab()
    sets = pr.degree_sets(P, vocab, int(ev["prompt_seed"]))
    common.write_json(sets, os.path.join(out, "e2_prompts_used.json"))
    if len(sets) <= 1:
        print("[warn] no complete degree sets in prompts.yaml; only the random control will run")

    idx = stats.bootstrap_indices(len(imgs), int(ev["bootstrap"]), int(ev["boot_seed"]))
    scale = float(ev["logit_scale"])
    text_rows, sp_rows, per_image = [], [], {"path": imgs}
    for name, levels in sets.items():
        L = np.stack([clip.class_embedding(lv) for lv in levels])  # (3, d)
        s, m, v = L
        axis = v - s
        t = float(np.dot(m - s, axis) / max(np.dot(axis, axis), 1e-12))
        text_rows.append(dict(set=name, t_moderate=t, ordered=bool(0 < t < 1),
                              cos_slight_moderate=float(s @ m), cos_moderate_severe=float(m @ v),
                              cos_slight_severe=float(s @ v)))
        probs = common.softmax(scale * img @ L.T)
        E = probs @ np.array([0.0, 1.0, 2.0])
        per_image[f"E_{name}"] = E
        for px in proxies:
            x = prox[px].to_numpy()
            rho, p, dist = stats.boot_spearman(E, x, idx)
            lo, hi = stats.ci(dist)
            sp_rows.append(dict(set=name, proxy=px, spearman=rho, ci_lo=lo, ci_hi=hi, p_value=p,
                                mean_E=float(E.mean()), sd_E=float(E.std())))

    pd.DataFrame(text_rows).to_csv(os.path.join(out, "e2_text_order.csv"), index=False)
    sp = pd.DataFrame(sp_rows)
    sp.to_csv(os.path.join(out, "e2_spearman.csv"), index=False)
    pi = pd.DataFrame(per_image).merge(prox, on="path")
    pi.to_csv(os.path.join(out, "e2_per_image.csv"), index=False)
    common.save_run_info(cfg, args.prompts, out, {"E2": {"n_images": len(imgs), "sets": list(sets)}})
    print(pd.DataFrame(text_rows).round(3).to_string(index=False))
    print(sp.pivot(index="set", columns="proxy", values="spearman").round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(sets)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.2), squeeze=False)
    for a, name in zip(axes[0], names):
        a.scatter(pi["area_ratio"], pi[f"E_{name}"], s=6, alpha=0.5)
        r = sp[(sp.set == name) & (sp.proxy == "area_ratio")].iloc[0]
        a.set_title(f"{name}\nrho={r.spearman:.2f} [{r.ci_lo:.2f},{r.ci_hi:.2f}]", fontsize=8)
        a.set_xlabel("crack area ratio (mask)", fontsize=8)
        a.set_ylabel("expected level E (0-2)", fontsize=8)
        a.set_xscale("symlog", linthresh=1e-3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "e2_scatter.png"), dpi=130)
    plt.close(fig)
    print(f"[done] E2 results in {out}")


if __name__ == "__main__":
    main()
