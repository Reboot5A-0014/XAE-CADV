#!/usr/bin/env bash
# XAE-CADV 训练 / 评估快捷脚本：训练 / 评估 / 分析（LUAD-HistoSeg 与 BCSS-WSSS）
#
# 用法:
#   ./run_member.sh {luad-train|luad-eval|bcss-train|bcss-eval|bcss-sweep|bcss-variants}
#   指定解释器：PY=/path/to/python ./run_member.sh luad-eval
#
# PY 需指向已装好 torch/numpy/mxnet 的环境（见 REPRO.md 第 0 节），默认 python3。
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
WT_INIT="init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params"
COMMON="--n_class 4 --img_size 224 --num_workers 8 --amp-dtype bf16"

case "${1:-}" in
  luad-train)
    "$PY" -u train_member.py --dataset luad --seed 11 --max_epoches 25 --batch_size 20 \
      --weights "$WT_INIT" --trainroot datasets/LUAD-HistoSeg/training \
      --save_folder checkpoints/luad_seed11 --eval_every 0 --save-last-k-checkpoints 0 $COMMON ;;
  luad-eval)
    "$PY" -u train_member.py --evaluate-only --dataset luad --seed 11 \
      --weights checkpoints/luad_seed11/stage1_last.pth \
      --testroot datasets/LUAD-HistoSeg/test $COMMON ;;
  bcss-train)
    "$PY" -u train_member.py --dataset bcss --seed 42 --max_epoches 25 --batch_size 20 \
      --weights "$WT_INIT" --trainroot datasets/BCSS-WSSS/training \
      --save_folder checkpoints/bcss_seed42 --eval_every 0 --save-last-k-checkpoints 0 $COMMON ;;
  bcss-eval)
    "$PY" -u train_member.py --evaluate-only --dataset bcss --seed 42 \
      --weights checkpoints/bcss_seed42/stage1_last.pth \
      --testroot datasets/BCSS-WSSS/test $COMMON ;;
  bcss-sweep)
    "$PY" -u sweep_infer.py --weights checkpoints/bcss_seed42/stage1_last.pth \
      --dataset bcss --testroot datasets/BCSS-WSSS/val $COMMON ;;
  bcss-variants)
    "$PY" -u infer_variants.py --weights checkpoints/bcss_seed42/stage1_last.pth \
      --tta-sets official,d4 --dataset bcss --testroot datasets/BCSS-WSSS/val $COMMON ;;
  *)
    echo "用法: $0 {luad-train|luad-eval|bcss-train|bcss-eval|bcss-sweep|bcss-variants}"; exit 1 ;;
esac
