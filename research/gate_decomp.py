"""门控缺陷分解：把"模型门控 vs ORACLE 门控"的差距拆成 假阳 / 假阴 两部分。

复用 softgate_sweep.py 的缓存（融合 CAM + 图像级概率），无需 GPU。
四种门控来源：
  pred          : (prob > thr)                     —— 官方
  gt            : mask 反推的存在性                 —— ORACLE
  gt ∪ pred     : 只补假阴（保留假阳）
  gt ∩ pred     : 只去假阳（保留假阴）
用法：
  python research/gate_decomp.py --cache /tmp/sg_bcss_val_seed42.npz --dataset bcss
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hist_iu_dice(hist):
    hist = hist.copy()
    hist[4, 4] = 0
    iu = np.diag(hist)[0:4] / ((hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))[0:4])
    mean_iu = float(np.nanmean(iu))
    dice = []
    for i in range(4):
        tp = np.diag(hist)[i]
        fp = hist[:, i].sum() - tp
        fn = hist[i, :].sum() - tp
        dice.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)
    return mean_iu, float(np.mean(dice)), iu


def evaluate(fused, labels_all, gts, n_class):
    hist = np.zeros((n_class + 1, n_class + 1))
    for i, gt in enumerate(gts):
        lab = labels_all[i]
        if lab.sum() == 0:
            pred = np.full(gt.shape, n_class, dtype=np.int64)
        else:
            cam = fused[i].astype(np.float32) * lab.reshape(n_class, 1, 1)
            pred = np.asarray(cam.argmax(axis=0), dtype=np.int64)
        gt_ = gt.astype(np.int64)
        pred = pred.copy()
        pred[gt_ == 4] = 4
        mask = (gt_ >= 0) & (gt_ < n_class + 1)
        b = (n_class + 1) * gt_[mask] + pred[mask]
        hist += np.bincount(b, minlength=(n_class + 1) ** 2).reshape(n_class + 1, n_class + 1)
    return hist_iu_dice(hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", default="bcss")
    ap.add_argument("--n_class", default=4, type=int)
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    fused, probs = z["fused"], z["probs"]
    gts = list(z["gts"])
    n_class = args.n_class

    if args.dataset == "bcss":
        thr = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
    else:
        thr = np.full(n_class, 0.2, dtype=np.float32)

    gt_pres = np.stack([[(g == c).any() for c in range(n_class)] for g in gts]).astype(np.float32)
    pred = (probs > thr).astype(np.float32)
    empty = pred.sum(axis=1) == 0
    if empty.any():
        pred[empty, np.argmax(probs[empty], axis=1)] = 1.0

    fn = ((gt_pres == 1) & (pred == 0)).sum()
    fp = ((gt_pres == 0) & (pred == 1)).sum()
    tp = ((gt_pres == 1) & (pred == 1)).sum()
    n_img = len(gts)
    print(f"实例级（图像×类）统计：TP={tp}  FN={fn}  FP={fp}   "
          f"平均每图 FN={fn/n_img:.3f}  FP={fp/n_img:.3f}")
    print(f"  → 漏检(FN) 占全部真阳的 {fn/(tp+fn)*100:.1f}%，误检(FP) 相对真阳 {fp/(tp+fn)*100:.1f}%\n")

    variants = {
        "pred（官方）": pred,
        "gt（ORACLE）": gt_pres,
        "gt ∪ pred（只补假阴）": np.maximum(gt_pres, pred),
        "gt ∩ pred（只去假阳）": np.minimum(gt_pres, pred),
    }
    base = None
    for name, lab in variants.items():
        lab = lab.copy()
        e = lab.sum(axis=1) == 0
        if e.any():
            lab[e, np.argmax(probs[e], axis=1)] = 1.0
        miou, mdice, iu = evaluate(fused, lab, gts, n_class)
        if base is None:
            base = miou
        print(f"{name:24s} mIoU={miou*100:6.2f}  mDice={mdice*100:6.2f}  ({((miou-base)*100):+.2f})  "
              f"clsIoU=[{iu[0]*100:.1f} {iu[1]*100:.1f} {iu[2]*100:.1f} {iu[3]*100:.1f}]")


if __name__ == "__main__":
    main()
