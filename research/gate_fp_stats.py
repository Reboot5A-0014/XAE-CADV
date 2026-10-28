"""假阳（FP）实例的可分性诊断：FP 与 TP 的空间形态是否可区分？

动机（gate_decomp.py 的一手证据）：BCSS val 上
  只去假阳（232 个实例） → mIoU +2.28
  只补假阴（1521 个实例） → mIoU +0.75
即**每个假阳的杀伤力约为假阴的 13 倍**。若 FP 在 CAM 空间形态上有可分性，
就可以设计一个"空间支撑否决"门控，在几乎不损失 TP 的前提下剔除 FP。

本脚本对每个 (图像, 被门控激活的类) 计算空间统计量，并按 GT 存在性分 TP/FP 两组，
报告每维特征的 AUC（TP 应为高分）以及直接网格组合的潜在收益上界。

用法：
  python research/gate_fp_stats.py --cache /tmp/sg_bcss_val_seed42.npz --dataset bcss
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def auc(score, label):
    """label=1 为正类。用秩统计实现（含并列的平均秩）。"""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s = score[order]
    i = 0
    r = np.arange(1, len(score) + 1, dtype=np.float64)
    # 处理并列
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = r[i:j + 1].mean()
        i = j + 1
    ranks[order] = r
    n1 = label.sum()
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return (ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def spatial_stats(cam_k):
    """输入单类归一化 CAM（[H,W]，min-max 后）。"""
    from scipy import ndimage
    hi = cam_k > 0.5
    mid = cam_k > 0.2
    area_hi = float(hi.mean())
    area_mid = float(mid.mean())
    if hi.any():
        lab, n = ndimage.label(hi)
        sizes = np.bincount(lab.ravel())[1:]
        max_cc = float(sizes.max() / hi.size)
        n_cc = float(n)
        # 最大连通域占自身高置信区面积的比例（碎片化程度）
        compact = float(sizes.max() / hi.sum())
    else:
        max_cc, n_cc, compact = 0.0, 0.0, 0.0
    p = cam_k.ravel().astype(np.float64)
    p = p / (p.sum() + 1e-12)
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size))
    top1 = float(np.sort(cam_k.ravel())[-max(1, int(0.01 * cam_k.size)):].mean())
    return dict(area_hi=area_hi, area_mid=area_mid, max_cc=max_cc, n_cc=n_cc,
                compact=compact, entropy=ent, top1=top1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", default="bcss")
    ap.add_argument("--n_class", default=4, type=int)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    fused, probs = z["fused"], z["probs"]
    gts = list(z["gts"])
    n_class = args.n_class
    thr = (np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32) if args.dataset == "bcss"
           else np.full(n_class, 0.2, dtype=np.float32))

    rows, labels = [], []
    for i, gt in enumerate(gts):
        cam = fused[i].astype(np.float32)
        lab = (probs[i] > thr).astype(np.float32)
        if lab.sum() == 0:
            lab[int(np.argmax(probs[i]))] = 1.0
        for k in np.where(lab > 0)[0]:
            st = spatial_stats(cam[k])
            st["prob"] = float(probs[i][k])
            st["cls"] = int(k)
            rows.append(st)
            labels.append(1.0 if (gt == k).any() else 0.0)
    labels = np.asarray(labels)

    keys = ["prob", "area_hi", "area_mid", "max_cc", "n_cc", "compact", "entropy", "top1"]
    A = np.stack([np.asarray([r[k] for r in rows]) for k in keys], axis=1)

    print(f"实例数 N={len(labels)}  TP={int(labels.sum())}  FP={int((1-labels).sum())}\n")
    print("%-10s %10s %10s %8s" % ("特征", "TP均值", "FP均值", "AUC"))
    for j, k in enumerate(keys):
        v = A[:, j]
        print("%-10s %10.4f %10.4f %8.3f" % (k, v[labels == 1].mean(), v[labels == 0].mean(),
                                             auc(v, labels)))

    # 简单否决规则的可达上界：按 prob 排序，找使 mIoU 最优的阈值（单类统一）
    print("\n--- 单特征否决扫描（否决该特征最低的 x% 实例，看 TP/FP 各损失多少）---")
    for j, k in enumerate(["prob", "area_hi", "area_mid", "max_cc", "compact", "top1", "entropy"]):
        v = A[:, keys.index(k)]
        qs = [0, 1, 2, 3, 5, 8, 12, 20]
        line = f"{k:10s}"
        for q in qs:
            if q == 0:
                line += f"  q0:FP=0,TP=0"
                continue
            t = np.percentile(v, q)
            veto = v <= t
            fp_cut = int(((labels == 0) & veto).sum())
            tp_cut = int(((labels == 1) & veto).sum())
            line += f"  q{q}:FP{-(fp_cut)}/TP{-(tp_cut)}"
        print(line)

    if args.out:
        np.savez(args.out, X=A, y=labels, keys=np.array(keys))


if __name__ == "__main__":
    main()
