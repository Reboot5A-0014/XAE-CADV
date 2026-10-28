#!/usr/bin/env bash
# HFRM 结构消融实验复现脚本（BL / GSR / CH15 / K7 / K15 / K21）
#
# 依次训练并评估 6 个配置：BL / +GSR / +CH(K=15) / 全模型 K=7 / K=15 / K=21
# 其中 K=15（全模型）直接复用 checkpoints/ 下已有的正式权重，不重复训练。
#
# 用法:
#   ./run_ablation.sh luad 11
#   ./run_ablation.sh bcss 42
#   ./run_ablation.sh luad 11 && ./run_ablation.sh bcss 42     # 两个数据集串行
#   指定解释器：PY=/path/to/python ./run_ablation.sh bcss 42
#
# 产物:
#   checkpoints_ablation/<ds>_<config>_seed<seed>/stage1_last.pth   训练权重
#   logs/ablation_<ds>_<config>_seed<seed>.log                      完整日志
#   ablation_results.csv                                            汇总结果
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
WT_INIT="init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params"
COMMON="--n_class 4 --img_size 224 --num_workers 8 --amp-dtype bf16"
RESULTS="ablation_results.csv"
LOGDIR="logs"
mkdir -p "$LOGDIR"

DS="${1:?用法: $0 <luad|bcss> <seed>}"
SEED="${2:?用法: $0 <luad|bcss> <seed>}"

if [ "$DS" = "luad" ]; then
  TRAINROOT="datasets/LUAD-HistoSeg/training"
  TESTROOT="datasets/LUAD-HistoSeg/test"
elif [ "$DS" = "bcss" ]; then
  TRAINROOT="datasets/BCSS-WSSS/training"
  TESTROOT="datasets/BCSS-WSSS/test"
else
  echo "未知数据集: $DS"; exit 1
fi

# 格式: 名称|额外参数|复用的已有权重（为空则训练）
CONFIGS=(
  "BL|--no-gsr --no-ch|"
  "GSR|--no-ch|"
  "CH15|--no-gsr|"
  "K7|--hfrm_kernel 7|"
  "K15||checkpoints/${DS}_seed${SEED}/stage1_last.pth"
  "K21|--hfrm_kernel 21|"
)

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name flags existing <<< "$entry"
  log="$LOGDIR/ablation_${DS}_${name}_seed${SEED}.log"
  ckpt="checkpoints_ablation/${DS}_${name}_seed${SEED}/stage1_last.pth"

  if [ -z "$existing" ]; then
    echo "=== [$(date '+%F %T')] 训练 ${DS}/${name} (seed ${SEED}) flags='${flags}' ===" | tee -a "$log"
    "$PY" -u train_member.py --dataset "$DS" --seed "$SEED" --max_epoches 25 --batch_size 20 \
      --weights "$WT_INIT" --trainroot "$TRAINROOT" \
      --save_folder "checkpoints_ablation/${DS}_${name}_seed${SEED}" \
      --eval_every 0 --save-last-k-checkpoints 0 $COMMON $flags >> "$log" 2>&1
    if [ ! -f "$ckpt" ]; then
      echo "!!! 训练未产出权重，跳过评估: $ckpt" | tee -a "$log"
      continue
    fi
  else
    ckpt="$existing"
    echo "=== [$(date '+%F %T')] 复用已有权重 ${DS}/${name}: $ckpt ===" | tee -a "$log"
  fi

  echo "=== [$(date '+%F %T')] 评估 ${DS}/${name} ($ckpt) ===" | tee -a "$log"
  "$PY" -u train_member.py --evaluate-only --dataset "$DS" --seed "$SEED" \
    --weights "$ckpt" --testroot "$TESTROOT" $COMMON $flags >> "$log" 2>&1

  "$PY" - "$RESULTS" "$DS" "$name" "$SEED" "$ckpt" "$log" <<'PYEOF'
import ast, csv, os, sys
res_csv, ds, name, seed, ckpt, log = sys.argv[1:7]
line = None
with open(log) as f:
    for l in f:
        if l.startswith('[test_tta]'):
            line = l
miou = mdice = ''
if line:
    d = ast.literal_eval(line.split('[test_tta]', 1)[1].strip())
    miou, mdice = d['Mean IoU'], d['Mean Dice']
new = not os.path.exists(res_csv)
with open(res_csv, 'a', newline='') as f:
    w = csv.writer(f)
    if new:
        w.writerow(['dataset', 'config', 'seed', 'mIoU', 'mDice', 'checkpoint'])
    w.writerow([ds, name, seed, miou, mdice, ckpt])
print('RESULT %s %s seed=%s mIoU=%s mDice=%s' % (ds, name, seed, miou, mdice))
PYEOF
done

echo "=== [$(date '+%F %T')] ${DS} 消融全部完成，汇总: $RESULTS ==="
