"""Run before the real experiments: checks prompts.yaml, data folders and the GPU."""
from prelim import common, datasets, prompts as pr


def main():
    ap = common.base_argparser(__doc__)
    args = ap.parse_args()
    cfg = common.load_config(args.config, args.set)
    P = common.load_yaml(args.prompts)
    n = int(cfg["eval"]["n_prompts"])
    ok = True

    print("== prompts.yaml")
    errors, warns = pr.check_prompts(P, n, cfg["eval"]["conditions"])
    for c, v in P["classes"].items():
        print(f"  {c:14s} label='{v.get('label')}' term_literal='{v.get('term_literal')}' "
              f"attr={len(v.get('attr') or [])} llm={len(v.get('llm') or [])}")
    for w in warns:
        print("  [warn]", w)
    for e in errors:
        print("  [error]", e)
    ok &= not errors
    sets = [k for k, s in P["degree"]["sets"].items()
            if len(s["levels"]) == 3 and not any("TODO" in p for lv in s["levels"] for p in lv)]
    print(f"  complete degree sets for E2: {sets}")

    print("== data")
    try:
        root = datasets.find_codebrim_root(cfg["data"]["codebrim_root"])
        for sp in ("val", "test"):
            try:
                paths, y = datasets.load_codebrim(root, sp)
                print(f"  CODEBRIM {sp}: {len(paths)} images; per class: "
                      + ", ".join(f"{t}={int(y[:, i].sum())}" for i, t in enumerate(datasets.CODEBRIM_TAGS)))
            except Exception as ex:  # noqa: BLE001
                print(f"  CODEBRIM {sp}: {ex}")
                ok &= sp != cfg["data"]["codebrim_split"]
    except Exception as ex:  # noqa: BLE001
        print("  [error]", ex)
        ok = False
    try:
        imgs, _ = datasets.load_deepcrack(cfg["data"]["deepcrack_root"], cfg["data"]["deepcrack_splits"])
        print(f"  DeepCrack: {len(imgs)} image/mask pairs")
    except Exception as ex:  # noqa: BLE001
        print("  [error]", ex)
        ok = False

    print("== runtime")
    import torch
    print(f"  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " -> will be SLOW on CPU"))
    print("\nREADY" if ok else "\nNOT READY - fix the [error] lines above")


if __name__ == "__main__":
    main()
