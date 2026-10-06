#!/usr/bin/env bash
# 填写完 observations.md 之后运行：重新生成报告，并把所有需要交回的文件打成一个 zip
set -uo pipefail
cd "$(dirname "$0")"
for spec in "vitl14_openai ViT-L-14-quickgelu" "vitb16_openai ViT-B-16-quickgelu"; do
  read -r RUN ARCH <<< "$spec"
  if [ -d "results/$RUN" ]; then python make_report.py --set model.arch=$ARCH run_name=$RUN; fi
done
python - <<'PY'
import glob, os, time, zipfile
out = f"results/send_to_claude_{time.strftime('%Y%m%d_%H%M')}.zip"
files = sorted(glob.glob("results/*_bundle.zip") + glob.glob("results/run_log_*.txt")
               + glob.glob("results/check_setup_*.txt")) + ["prompts.yaml"]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f):
            z.write(f)
print("已打包：", *files, sep="\n  ")
print(f"\n请把这个文件发给 Claude：{out}  ({os.path.getsize(out)/1e6:.1f} MB)")
PY
