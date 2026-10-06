#!/usr/bin/env bash
# 正式运行：两个主干网络 × E1/E2/E3 + 报告。全部输出同时写入 results/run_log_*.txt
# 用法：bash run_all.sh
#       显存不足：EXTRA="runtime.batch_size=16" bash run_all.sh
#       DataLoader 报错：EXTRA="runtime.num_workers=0" bash run_all.sh
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p results
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/run_log_${STAMP}.txt"
EXTRA="${EXTRA:-}"
exec > >(tee -a "$LOG") 2>&1

echo "== start $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "[warn] nvidia-smi 不可用"
python -c "import torch, open_clip; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'open_clip', open_clip.__version__)"
python check_setup.py | tee "results/check_setup_${STAMP}.txt"
if ! grep -q "^READY" "results/check_setup_${STAMP}.txt"; then
  echo "[stop] check_setup 没有显示 READY，请先修正上面的 [error]"; exit 1
fi

step() {
  echo; echo "== $* :: start $(date +%T)"; SECONDS=0
  "$@"; rc=$?
  echo "== $* :: exit=$rc, ${SECONDS}s"
  if [ $rc -ne 0 ]; then echo "[stop] 出错。请把 $LOG 发给 Claude"; exit $rc; fi
}

for spec in "vitl14_openai ViT-L-14-quickgelu" "vitb16_openai ViT-B-16-quickgelu"; do
  read -r RUN ARCH <<< "$spec"
  S="model.arch=$ARCH run_name=$RUN $EXTRA"
  step python e1_codebrim.py --set $S
  step python e2_ordinal.py --set $S
  step python e3_localization.py --set $S
  step python make_report.py --set $S
done
echo; echo "== all done $(date)"
echo "下一步：查看 results/*/summary.md 和 figures/，填写 observations.md，然后运行 bash pack_results.sh"
