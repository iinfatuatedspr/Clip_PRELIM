"""Pipeline self-test (no downloads, no pretrained weights, runs on CPU in a few minutes).

Creates tiny fake datasets with the same folder layout as CODEBRIM / DeepCrack
and runs E1-E3 + the report with a randomly initialised ViT-B-32.
The numbers it prints are meaningless; it only proves the code runs.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.join("data", "_selftest")
TAGS = ["Background", "Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain"]


def fake_codebrim(rng):
    base = os.path.join(ROOT, "codebrim", "CODEBRIM_classification_dataset")
    os.makedirs(os.path.join(base, "metadata"), exist_ok=True)
    xml = {"background": [], "defects": []}
    for split, n in (("val", 12), ("test", 30)):
        for sub in ("background", "defects"):
            d = os.path.join(base, split, sub)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                name = f"{split}_{sub}_{i:04d}.png"
                w, h = rng.integers(120, 400, size=2)
                arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(d, name))
                if sub == "background":
                    vec = [1, 0, 0, 0, 0, 0]
                else:
                    vec = [0] + list(rng.integers(0, 2, size=5))
                    if sum(vec) == 0:
                        vec[1 + i % 5] = 1
                xml[sub].append((name, vec))
    for sub, items in xml.items():
        lines = ["<Defects>"]
        for name, vec in items:
            lines.append(f'  <Defect name="{name}">' + "".join(f"<{t}>{v}</{t}>" for t, v in zip(TAGS, vec))
                         + "</Defect>")
        lines.append("</Defects>")
        with open(os.path.join(base, "metadata", f"{sub}.xml"), "w") as f:
            f.write("\n".join(lines))


def fake_deepcrack(rng):
    base = os.path.join(ROOT, "deepcrack", "DeepCrack")
    for split, n in (("train", 8), ("test", 8)):
        os.makedirs(os.path.join(base, f"{split}_img"), exist_ok=True)
        os.makedirs(os.path.join(base, f"{split}_lab"), exist_ok=True)
        for i in range(n):
            img = Image.fromarray((rng.random((256, 192, 3)) * 80 + 120).astype(np.uint8))
            mask = Image.new("L", (192, 256), 0)
            d1, d2 = ImageDraw.Draw(img), ImageDraw.Draw(mask)
            wid = int(rng.integers(1, 9))
            pts = [tuple(int(v) for v in rng.integers(0, 190, size=2)) for _ in range(3)]
            d1.line(pts, fill=(20, 20, 20), width=wid)
            d2.line(pts, fill=255, width=wid)
            img.save(os.path.join(base, f"{split}_img", f"{i:03d}.jpg"))
            mask.save(os.path.join(base, f"{split}_lab", f"{i:03d}.png"))


def filled_prompts():
    """Copy prompts.yaml and fill every TODO with dummy text so that all code paths run."""
    import yaml
    P = yaml.safe_load(open("prompts.yaml", encoding="utf-8"))
    for name, c in P["classes"].items():
        c["term_literal"] = f"dummy term {name}"
        c["attr"] = [{"text": f"dummy visual attribute {i}", "source": "selftest"} for i in range(10)]
        c["llm"] = [f"dummy paraphrase {i}" for i in range(12)]
    P["degree"]["sets"]["manual"]["levels"] = [[f"level {k} phrase {i}" for i in range(3)] for k in range(3)]
    path = os.path.join(ROOT, "prompts_filled.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(P, f, allow_unicode=True, sort_keys=False)
    return path


def run(cmd):
    print("\n$", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"SELFTEST FAILED at: {' '.join(cmd)}")


def main():
    rng = np.random.default_rng(0)
    fake_codebrim(rng)
    fake_deepcrack(rng)
    prompts = filled_prompts()
    common = ["--prompts", prompts, "--set", "run_name=selftest", "model.arch=ViT-B-32", "model.pretrained=null",
              f"data.codebrim_root={ROOT}/codebrim", f"data.deepcrack_root={ROOT}/deepcrack",
              "eval.bootstrap=50", "e3.bootstrap=20", "e3.max_pixels_per_image=500", "e3.n_vis=2",
              "eval.n_vis=2", "paths.results_dir=results_selftest"]
    py = sys.executable
    run([py, "check_setup.py"] + common)
    run([py, "e1_codebrim.py"] + common)
    run([py, "e1_codebrim.py", "--split", "val", "--limit", "10", "--no_vis"]
        + [x if not x.startswith("run_name=") else "run_name=selftest_smoke" for x in common])
    run([py, "e2_ordinal.py"] + common)
    run([py, "e3_localization.py"] + common)
    run([py, "make_report.py"] + common)
    print("\nSELFTEST OK - the pipeline runs end to end (numbers above are meaningless: random weights).")


if __name__ == "__main__":
    main()
