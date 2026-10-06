"""E0 (optional) - How often do the damage terms appear in image-caption data?

Tests the premise of research question (1): are the inspection terms really
rare in the kind of text CLIP-style models are trained on?
Counts the number of captions containing each term (case-insensitive, whole
words) in a TSV caption file such as CC12M (url<TAB>caption).

    python e0_caption_freq.py --captions data/cc12m.tsv --col 1
"""
import argparse
import os
import re

import pandas as pd

from prelim import common


def terms_from_prompts(P):
    terms = set()
    for c in P["classes"].values():
        for key in ("label", "term_literal"):
            t = str(c.get(key, ""))
            if t and "TODO" not in t:
                terms.add(re.sub(r"^(a|an|the)\s+", "", t.strip().lower()))
    for t in P.get("extra_frequency_terms", []) or []:
        terms.add(str(t).lower())
    return sorted(terms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--col", type=int, default=1, help="0-based column index of the caption in the TSV")
    ap.add_argument("--max_lines", type=int, default=None)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--prompts", default="prompts.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = common.load_config(args.config, args.set)
    P = common.load_yaml(args.prompts)
    out = common.run_dir(cfg)

    terms = terms_from_prompts(P)
    pats = {t: re.compile(r"\b" + re.escape(t) + r"\b") for t in terms}
    first = {t: t.split()[0] for t in terms}
    counts = {t: 0 for t in terms}
    n = 0
    from tqdm.auto import tqdm
    with open(args.captions, encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="captions", unit_scale=True):
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= args.col:
                continue
            cap = parts[args.col].lower()
            n += 1
            for t in terms:
                if first[t] in cap and pats[t].search(cap):
                    counts[t] += 1
            if args.max_lines and n >= args.max_lines:
                break
    df = pd.DataFrame([{"term": t, "captions_with_term": c, "per_million": c / max(n, 1) * 1e6}
                       for t, c in counts.items()]).sort_values("per_million", ascending=False)
    df["corpus"] = os.path.basename(args.captions)
    df["n_captions"] = n
    df.to_csv(os.path.join(out, "e0_caption_freq.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
