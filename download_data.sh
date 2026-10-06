#!/usr/bin/env bash
# 下载并解压实验数据（CODEBRIM 只解压 metadata/val/test；DeepCrack 全部）。可重复运行，已完成的部分会跳过。
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p data/codebrim data/deepcrack downloads
URL="https://zenodo.org/records/2620293/files/CODEBRIM_classification_dataset.zip?download=1"
MD5="c1612d9674e2e628e72e7f5817c40130"

if [ -f downloads/.codebrim_ok ]; then
  echo "[skip] CODEBRIM 已解压"
else
  echo "== 下载 CODEBRIM（7.9 GB，支持断点续传：中断后重新运行本脚本即可）"
  wget -c --show-progress -O downloads/codebrim.zip "$URL" || { echo "[error] 下载失败，请重新运行"; exit 1; }
  echo "== 校验 md5"
  echo "$MD5  downloads/codebrim.zip" | md5sum -c - || { echo "[error] md5 不一致：删除 downloads/codebrim.zip 后重新运行"; exit 1; }
  echo "== 解压 metadata / val / test"
  unzip -q -o downloads/codebrim.zip '*metadata/*' '*val/*' '*test/*' -d data/codebrim
  rc=$?; [ $rc -ne 0 ] && echo "[warn] unzip 退出码 $rc，请检查下面的目录列表"
  touch downloads/.codebrim_ok
fi

if [ -d data/deepcrack/test_img ]; then
  echo "[skip] DeepCrack 已解压"
else
  echo "== 下载 DeepCrack"
  [ -d downloads/DeepCrack_repo ] || git clone -q --depth 1 https://github.com/yhlleo/DeepCrack.git downloads/DeepCrack_repo
  unzip -q -o downloads/DeepCrack_repo/dataset/DeepCrack.zip -d data/deepcrack
fi

echo "== CODEBRIM 目录（应有 metadata、val、test，val/test 下有 background、defects）"
find data/codebrim -maxdepth 3 -type d
echo "== DeepCrack 目录（应有 test_img test_lab train_img train_lab）"
ls data/deepcrack
echo "== 磁盘剩余"; df -h . | tail -1
echo "确认无误后，可执行 rm downloads/codebrim.zip 释放约 8 GB 空间"
