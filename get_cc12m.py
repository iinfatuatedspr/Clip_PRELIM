"""Download CC12M captions (text only) from Hugging Face and write downloads/cc12m.tsv (url<TAB>caption)."""
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "google-research-datasets/conceptual_12m"
files, revision = None, None
for rev in ["main", "refs/convert/parquet"]:
    try:
        fs = sorted(f for f in list_repo_files(REPO, repo_type="dataset", revision=rev) if f.endswith(".parquet"))
        if fs:
            files, revision = fs, rev
            break
    except Exception as e:  # noqa: BLE001
        print(f"[info] revision {rev}: {e}")
if not files:
    raise SystemExit("no parquet files found on Hugging Face")
print("revision:", revision, "| files:", files)

n = 0
with open("downloads/cc12m.tsv", "w", encoding="utf-8") as out:
    for f in files:
        p = hf_hub_download(REPO, f, repo_type="dataset", revision=revision, local_dir="downloads/cc12m_parquet")
        df = pd.read_parquet(p)
        cap = "caption" if "caption" in df.columns else [c for c in df.columns if "caption" in c.lower()][0]
        url = "image_url" if "image_url" in df.columns else df.columns[0]
        for u, c in zip(df[url], df[cap]):
            c = str(c).replace("\t", " ").replace("\n", " ")
            out.write(f"{u}\t{c}\n")
            n += 1
        print(f"{f}: {len(df)} rows, total {n}")
print("rows written:", n)
