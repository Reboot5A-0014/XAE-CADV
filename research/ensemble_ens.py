"""Phase-1 多模型集成（门控概率集成 + CAM 集成）。

动机：门控缺陷分解显示 BCSS val 上"只去假阳"值 +2.28 mIoU，是当前最大的可兑现缺口；
而集成图像级概率是降低门控方差/假阳的标准手段（此前只做过"CAM 平均"，未做"概率集成 + 阈值重标定"）。

流程：
  1) 逐个模型收集 val / test 的 融合CAM + 图像级概率（缓存到 /tmp/ens/）
  2) 增量累加得到集成量（避免一次性载入全部缓存导致 OOM）
  3) 在 val 上做逐类阈值坐标搜索，报 test 指标
  4) 同时给出单模型与若干子集的对照

用法：
  python research/ensemble_ens.py --dataset bcss --stage collect
  python research/ensemble_ens.py --dataset bcss --stage aggregate
"""
import argparse
import importlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.softgate_sweep import collect, hist_iu_dice  # noqa: E402

CACHE = "/tmp/ens"

MODELS = {
    "bcss": [
        # tag, weights, gsr, ch, kernel
        ("k15_s42", "checkpoints/bcss_seed42/stage1_last.pth", True, True, 15),
        ("k15_s11", "checkpoints_ms/bcss_s11/stage1_last.pth", True, True, 15),
        ("k15_s21", "checkpoints_ms/bcss_s21/stage1_last.pth", True, True, 15),
        ("k15_s33", "checkpoints_ms/bcss_s33/stage1_last.pth", True, True, 15),
        ("k15_s44", "checkpoints_ms/bcss_s44/stage1_last.pth", True, True, 15),
        ("k21_s42", "checkpoints_ablation/bcss_K21_seed42/stage1_last.pth", True, True, 21),
        ("aug_s42", "checkpoints_aug/bcss_aug/stage1_last.pth", True, True, 15),
        ("bl_s42", "checkpoints_ablation/bcss_BL_seed42/stage1_last.pth", False, False, 15),
        ("gsr_s42", "checkpoints_ablation/bcss_GSR_seed42/stage1_last.pth", True, False, 15),
        ("ch15_s42", "checkpoints_ablation/bcss_CH15_seed42/stage1_last.pth", False, True, 15),
        ("k7_s42", "checkpoints_ablation/bcss_K7_seed42/stage1_last.pth", True, True, 7),
        # 以下由 research/collect_fm.py 预先写入缓存（此处只声明 tag，collect 阶段会因权重不存在而跳过）
        ("pv2", "", True, True, 15),
    ],
    "luad": [
        ("k15_s11", "checkpoints/luad_seed11/stage1_last.pth", True, True, 15),
        ("k15_s42", "checkpoints_ms/luad_s42/stage1_last.pth", True, True, 15),
        ("k15_s21", "checkpoints_ms/luad_s21/stage1_last.pth", True, True, 15),
        ("k15_s33", "checkpoints_ms/luad_s33/stage1_last.pth", True, True, 15),
        ("k15_s44", "checkpoints_ms/luad_s44/stage1_last.pth", True, True, 15),
        ("bl_s11", "checkpoints_ablation/luad_BL_seed11/stage1_last.pth", False, False, 15),
        ("gsr_s11", "checkpoints_ablation/luad_GSR_seed11/stage1_last.pth", True, False, 15),
        ("ch15_s11", "checkpoints_ablation/luad_CH15_seed11/stage1_last.pth", False, True, 15),
        ("k7_s11", "checkpoints_ablation/luad_K7_seed11/stage1_last.pth", True, True, 7),
        ("k21_s11", "checkpoints_ablation/luad_K21_seed11/stage1_last.pth", True, True, 21),
        ("k15_s45", "checkpoints_p2fm/luad_s45/stage1_last.pth", True, True, 15),
        ("pv2", "", True, True, 15),
    ],
}


def paths(args, split):
    root = os.path.join("datasets", "BCSS-WSSS" if args.dataset == "bcss" else "LUAD-HistoSeg")
    return os.path.join(root, split)


def do_collect(args):
    os.makedirs(CACHE, exist_ok=True)
    for tag, w, gsr, ch, k in MODELS[args.dataset]:
        if not os.path.exists(w):
            print(f"[skip] {tag}: 权重不存在 {w}")
            continue
        model = getattr(importlib.import_module("network.resnet38_cls"), "Net_CAM")(
            n_class=args.n_class, hfrm_gsr=gsr, hfrm_ch=ch, hfrm_kernel=k)
        model.load_state_dict(torch.load(w), strict=False)
        for split in ("val", "test"):
            out = f"{CACHE}/{args.dataset}_{tag}_{split}.npz"
            if os.path.exists(out):
                print(f"[cached] {out}")
                continue
            fused, probs, gts = collect(model, paths(args, split), args, args.n_class, args.weights_tuple)
            np.savez(out, fused=fused, probs=probs, gts=np.array(gts, dtype=object))
            print(f"[saved] {out}", flush=True)


def _accumulate(args, tags, split):
    """增量累加：返回 (probs_sum, cam_sum, gts)。"""
    ps = cs = None
    gts = None
    for tag in tags:
        f = f"{CACHE}/{args.dataset}_{tag}_{split}.npz"
        if not os.path.exists(f):
            print(f"[warn] 缺缓存 {f}")
            continue
        z = np.load(f, allow_pickle=True)
        if ps is None:
            ps = z["probs"].astype(np.float64)
            cs = z["fused"].astype(np.float32)
            gts = list(z["gts"])
        else:
            ps += z["probs"]
            cs += z["fused"].astype(np.float32)
    return ps, cs, gts


def labels_from(probs, thr):
    lab = (probs > thr).astype(np.float32)
    e = lab.sum(axis=1) == 0
    if e.any():
        lab[e, np.argmax(probs[e], axis=1)] = 1.0
    return lab


def evaluate(probs, cams, gts, thr, n_class):
    """向量化评估（与 tool/iouutils.scores 同口径）。"""
    lab_all = labels_from(probs, thr)
    N = len(gts)
    hist = np.zeros((n_class + 1, n_class + 1))
    B = 128
    for st in range(0, N, B):
        en = min(st + B, N)
        cam = cams[st:en].astype(np.float32) * lab_all[st:en, :, None, None]
        pred = cam.argmax(axis=1).astype(np.int64)
        g = np.stack([gts[i] for i in range(st, en)]).astype(np.int64)
        pred = pred.copy()
        pred[g == n_class] = n_class
        b = (n_class + 1) * g.ravel() + pred.ravel()
        hist += np.bincount(b, minlength=(n_class + 1) ** 2).reshape(n_class + 1, n_class + 1)
    return hist_iu_dice(hist)


def tune(probs, cams, gts, n_class, default_thr, rounds=2, grid=None):
    if grid is None:
        grid = np.round(np.arange(0.30, 0.96, 0.05), 2)
    thr = default_thr.copy()
    best = evaluate(probs, cams, gts, thr, n_class)[0]
    for _ in range(rounds):
        improved = False
        for c in range(n_class):
            for t in grid:
                cand = thr.copy()
                cand[c] = t
                s = evaluate(probs, cams, gts, cand, n_class)[0]
                if s > best + 1e-6:
                    best, thr, improved = s, cand, True
        if not improved:
            break
    return best, thr


def do_aggregate(args):
    n = args.n_class
    default_thr = (np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32) if args.dataset == "bcss"
                   else np.full(n, 0.2, dtype=np.float32))
    tags = [m[0] for m in MODELS[args.dataset] if os.path.exists(f"{CACHE}/{args.dataset}_{m[0]}_test.npz")]
    print(f"参与集成的模型（{len(tags)}）：{tags}\n")

    # 单模型对照
    print("--- 单模型（test，默认阈值）---")
    best_single = None
    for tag in tags:
        tp, tc, tg = _accumulate(args, [tag], "test")
        m, d, _ = evaluate(tp, tc, tg, default_thr, n)
        print(f"  {tag:10s} test mIoU={m*100:6.2f}  mDice={d*100:6.2f}")
        if best_single is None or m > best_single[0]:
            best_single = (m, tag)
        del tp, tc
    print(f"  → 单模型最佳：{best_single[1]} {best_single[0]*100:.2f}\n")

    # 集成（概率平均 + CAM 平均）
    print("--- 集成 ---")
    for name, sub in [("全集成", tags), ("去BL", [t for t in tags if t != "bl_s42"]),
                      ("仅K15五seed", [t for t in tags if t.startswith("k15_")])]:
        if len(sub) < 2:
            continue
        vps, vcs, vgs = _accumulate(args, sub, "val")
        tps, tcs, tgs = _accumulate(args, sub, "test")
        vps /= len(sub); tps /= len(sub); vcs /= len(sub); tcs /= len(sub)
        m0, d0, _ = evaluate(tps, tcs, tgs, default_thr, n)
        bv, bthr = tune(vps, vcs, vgs, n, default_thr)
        mt, dt, iu = evaluate(tps, tcs, tgs, bthr, n)
        print(f"  {name:12s} 默认thr test {m0*100:6.2f} | 调优thr {np.round(bthr,2).tolist()} "
              f"val {bv*100:6.2f} → test {mt*100:6.2f} ({((mt-m0)*100):+.2f})  mDice {dt*100:6.2f}")
        # 只集成概率门控，CAM 用最佳单模型
        _, bc, _ = _accumulate(args, [best_single[1]], "test")
        mb, db, _ = evaluate(tps, bc, tgs, bthr, n)
        print(f"  {'':12s} └ 仅概率集成+最佳单模型CAM：test {mb*100:6.2f}  mDice {db*100:6.2f}")
        del vps, vcs, tps, tcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bcss")
    ap.add_argument("--stage", default="collect", choices=["collect", "aggregate"])
    ap.add_argument("--n_class", default=4, type=int)
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--num_workers", default=8, type=int)
    ap.add_argument("--amp_dtype", "--amp-dtype", default="bf16")
    ap.add_argument("--cam_w", default="0.6,0.2,0.2")
    args = ap.parse_args()
    args.weights_tuple = tuple(float(x) for x in args.cam_w.split(","))
    if args.stage == "collect":
        do_collect(args)
    else:
        do_aggregate(args)


if __name__ == "__main__":
    main()
