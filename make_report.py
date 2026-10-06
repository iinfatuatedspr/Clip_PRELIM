"""Collect all result CSVs of one run into summary.md and a zip bundle to send back."""
import json
import os
import shutil
import zipfile

import pandas as pd

from prelim import common, stats


def md_table(df, floatfmt=3):
    if df is None or df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.{floatfmt}f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def read(out, name):
    p = os.path.join(out, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    ap = common.base_argparser(__doc__)
    args = ap.parse_args()
    cfg = common.load_config(args.config, args.set)
    out = common.run_dir(cfg)
    shutil.copy(args.prompts, os.path.join(out, "prompts_snapshot.yaml"))
    info = {}
    if os.path.exists(os.path.join(out, "run_info.json")):
        info = json.load(open(os.path.join(out, "run_info.json"), encoding="utf-8"))

    S = [f"# Preliminary experiment report: `{cfg['run_name']}`\n",
         f"- model: `{cfg['model']['arch']}` / `{cfg['model']['pretrained']}`",
         f"- gpu: {info.get('gpu')}, torch {info.get('torch')}, open_clip {info.get('open_clip')}",
         f"- prompts.yaml sha256: `{info.get('prompts_sha256', '')[:16]}`",
         f"- updated: {info.get('updated')}",
         f"- experiments: {json.dumps(info.get('experiments', {}), ensure_ascii=False)}\n"]

    e1 = read(out, "e1_auroc.csv")
    if e1 is not None:
        S.append("## E1 CODEBRIM: AUROC [95% bootstrap CI]\n")
        for nset in e1.normal_set.unique():
            d = e1[e1.normal_set == nset].copy()
            d["cell"] = [stats.fmt_ci(a, l, h) for a, l, h in zip(d.auroc, d.ci_lo, d.ci_hi)]
            order = ["T1_anomaly", "macro_T2"] + [t for t in d.target.unique() if t not in ("T1_anomaly", "macro_T2")]
            tab = d.pivot(index="condition", columns="target", values="cell")[order].reset_index()
            S.append(f"### normal prompts = `{nset}`\n")
            S.append(md_table(tab))
        pw = read(out, "e1_pairwise.csv")
        if pw is not None:
            S.append("### E1 paired differences (A - B), main targets\n")
            d = pw[pw.target.isin(["T1_anomaly", "macro_T2"])].copy()
            d["diff [CI]"] = [stats.fmt_ci(a, l, h) for a, l, h in zip(d["diff"], d.ci_lo, d.ci_hi)]
            S.append(md_table(d[["normal_set", "A", "B", "question", "target", "diff [CI]", "p_boot"]]))
        ns = read(out, "e1_normal_sets.csv")
        if ns is not None:
            S.append("### E1 hard-negative normal prompts (extended - basic), T1 anomaly\n")
            S.append(md_table(ns))
        cm = read(out, "e1_confusion.csv")
        if cm is not None:
            vc = cfg["eval"].get("vis_condition", "label_attr")
            d = cm[(cm.condition == vc) & (cm.normal_set == cfg["eval"]["normal_sets"][0])]
            S.append(f"### E1 confusion on single-label defect images (condition `{vc}`; rows=true, cols=top-1)\n")
            S.append(md_table(d.drop(columns=["normal_set", "condition"])))

    t2, sp = read(out, "e2_text_order.csv"), read(out, "e2_spearman.csv")
    if t2 is not None:
        S.append("## E2 DeepCrack: ordinality of degree expressions\n")
        S.append("Text side (`t_moderate` in (0,1) means 'moderate' lies between 'slight' and 'severe'):\n")
        S.append(md_table(t2))
        d = sp.copy()
        d["rho [CI]"] = [stats.fmt_ci(a, l, h) for a, l, h in zip(d.spearman, d.ci_lo, d.ci_hi)]
        S.append("Image side (Spearman between expected level and mask proxies):\n")
        S.append(md_table(d.pivot(index="set", columns="proxy", values="rho [CI]").reset_index()))

    e3 = read(out, "e3_pixel.csv")
    if e3 is not None:
        S.append("## E3 DeepCrack: window-based localisation\n")
        S.append(md_table(e3))
        S.append(md_table(read(out, "e3_pairwise.csv")))

    e0 = read(out, "e0_caption_freq.csv")
    if e0 is not None:
        S.append("## E0 term frequency in captions\n")
        S.append(md_table(e0))

    figs = sorted(os.listdir(os.path.join(out, "figures")))
    S.append("## Figures\n" + "\n".join(f"- figures/{f}" for f in figs) + "\n")
    obs = os.path.join(out, "observations.md")
    if not os.path.exists(obs):
        shutil.copy("observations_template.md", obs)
    S.append("## Observations\nSee `observations.md` (written by the experimenter).\n")

    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(S))

    bundle = os.path.join(cfg["paths"]["results_dir"], f"{cfg['run_name']}_bundle.zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out):
            for fn in files:
                if fn == "e1_scores.csv.gz" and os.path.getsize(os.path.join(root, fn)) > 20e6:
                    continue
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, cfg["paths"]["results_dir"]))
    print(f"[done] {os.path.join(out, 'summary.md')}\n[done] bundle: {bundle}")


if __name__ == "__main__":
    main()
