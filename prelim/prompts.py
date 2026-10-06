"""Turns prompts.yaml into the exact prompt lists used by each condition.

Conditions (every condition gets exactly `n_prompts` prompts per class):
  label        : template x English dataset label            ("a photo of spalling on ...")
  term_literal : template x machine translation of the inspection-manual term
  attr         : hand-written visual-attribute descriptions, no class name
  label_attr   : class label + the same visual-attribute descriptions
  llm          : LLM paraphrases of the attribute descriptions, no class name
  label_llm    : class label + LLM paraphrases
  waffle       : class label + random words / random characters (WaffleCLIP-style control)
"""
import string

import numpy as np

ALL_CONDITIONS = ["label", "term_literal", "attr", "label_attr", "llm", "label_llm", "waffle"]

# words that must never appear in the random control, because they carry damage semantics
_STOP = {
    "crack", "cracks", "cracked", "cracking", "rust", "rusty", "stain", "stains", "stained", "damage",
    "damaged", "broken", "break", "leak", "leaking", "water", "wet", "white", "brown", "orange", "red",
    "dark", "spall", "spalling", "concrete", "cement", "rebar", "steel", "iron", "metal", "exposed",
    "corrosion", "corroded", "salt", "lime", "mineral", "peel", "peeling", "chip", "chipped", "hole",
    "holes", "bar", "bars", "line", "lines", "defect", "defects", "wall", "bridge", "surface", "powder",
    "efflorescence", "deposit", "deposits", "flake", "flaking", "decay", "rot", "rotten", "fracture",
    "split", "gap", "scratch", "scratches", "dirt", "dirty", "mold", "moss", "shadow", "graffiti",
}


def _text(item):
    return item["text"] if isinstance(item, dict) else str(item)


def _source(item):
    return item.get("source", "") if isinstance(item, dict) else ""


def _is_todo(x):
    return isinstance(x, str) and "TODO" in x


def random_vocab():
    """Frequent, purely alphabetic words from CLIP's own BPE vocabulary (offline, deterministic)."""
    from open_clip.tokenizer import SimpleTokenizer

    tok = SimpleTokenizer()
    items = sorted(tok.encoder.items(), key=lambda kv: kv[1])
    words = []
    for w, i in items:
        if i >= 20000 or not w.endswith("</w>"):
            continue
        w = w[:-4]
        if w.isascii() and w.isalpha() and 4 <= len(w) <= 9 and w not in _STOP:
            words.append(w)
    return words


def check_prompts(P, n, conditions):
    """Return (errors, warnings). Errors block a real run unless --allow_incomplete."""
    errors, warns = [], []
    if len(P.get("templates", [])) < n:
        errors.append(f"templates: need at least {n}, found {len(P.get('templates', []))}")
    for cname, c in P["classes"].items():
        for key in ("label", "term_literal"):
            if _is_todo(c.get(key)):
                errors.append(f"classes.{cname}.{key} is still TODO")
        need = {"attr": "attr", "label_attr": "attr", "llm": "llm", "label_llm": "llm"}
        for cond in conditions:
            field = need.get(cond)
            if not field:
                continue
            items = [x for x in (c.get(field) or []) if not _is_todo(_text(x))]
            if len(items) < n:
                errors.append(f"classes.{cname}.{field}: need {n} items for condition '{cond}', found {len(items)}")
        for x in c.get("attr") or []:
            if "DRAFT" in _source(x):
                errors.append(f"classes.{cname}.attr still contains DRAFT example items (rewrite them from "
                              f"the inspection manual and fill in 'source')")
            if not _is_todo(_text(x)) and not _source(x):
                warns.append(f"classes.{cname}.attr item without source: '{_text(x)[:50]}'")
            if cname in _text(x).lower() or c.get("label", "").lower() in _text(x).lower():
                warns.append(f"classes.{cname}.attr item contains the class name: '{_text(x)[:50]}'")
    for sub, items in P["normal"]["extended"].items():
        if not items:
            errors.append(f"normal.extended.{sub} is empty")
    return sorted(set(errors)), sorted(set(warns))


def _pick(items, n, rng):
    items = [x for x in items if not _is_todo(_text(x))]
    if len(items) <= n:
        return list(items)
    idx = sorted(rng.choice(len(items), size=n, replace=False))
    return [items[i] for i in idx]


def build_conditions(P, conditions, n, seed, vocab):
    """{condition: {class: [prompt strings]}} with exactly n prompts per class when possible."""
    rng = np.random.default_rng(seed)
    templates = P["templates"][:n]
    t_desc = P["desc_template"]
    t_label_desc = P["label_desc_template"]
    out = {}
    for cond in conditions:
        out[cond] = {}
        for cname, c in P["classes"].items():
            label = c["label"]
            if cond == "label":
                prompts = [t.format(label) for t in templates]
            elif cond == "term_literal":
                prompts = [t.format(c["term_literal"]) for t in templates]
            elif cond in ("attr", "llm"):
                field = "attr" if cond == "attr" else "llm"
                prompts = [t_desc.format(_text(x)) for x in _pick(c.get(field) or [], n, rng)]
            elif cond in ("label_attr", "label_llm"):
                field = "attr" if cond == "label_attr" else "llm"
                prompts = [t_label_desc.format(label=label, desc=_text(x))
                           for x in _pick(c.get(field) or [], n, rng)]
            elif cond == "waffle":
                prompts = []
                for k in range(n):
                    if k % 2 == 0:
                        words = rng.choice(vocab, size=2, replace=False)
                        desc = f"{words[0]} and {words[1]}"
                    else:
                        letters = list(string.ascii_lowercase)
                        desc = " ".join("".join(rng.choice(letters, size=5)) for _ in range(2))
                    prompts.append(t_label_desc.format(label=label, desc=desc))
            else:
                raise ValueError(f"unknown condition {cond}")
            out[cond][cname] = prompts
    return out


def normal_sets(P, which):
    """{subclass: [prompts]}. 'basic' is one subclass; 'extended' adds hard negatives."""
    if which == "basic":
        return {"plain": list(P["normal"]["basic"])}
    if which == "extended":
        d = {"plain": list(P["normal"]["basic"])}
        d.update({k: list(v) for k, v in P["normal"]["extended"].items()})
        return d
    raise ValueError(which)


def degree_sets(P, vocab, seed):
    """Degree-wording sets for E2, plus an automatically generated random-order control."""
    tmpl = P["degree"]["template"]
    sets = {}
    for name, s in P["degree"]["sets"].items():
        levels = s["levels"]
        if any(_is_todo(p) for lv in levels for p in lv) or len(levels) != 3:
            continue
        sets[name] = [[tmpl.format(p) for p in lv] for lv in levels]
    rng = np.random.default_rng(seed + 17)
    words = rng.choice(vocab, size=9, replace=False)
    sets["random_control"] = [[tmpl.format(f"a {w} crack") for w in words[i * 3:(i + 1) * 3]] for i in range(3)]
    return sets
