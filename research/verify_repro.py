"""复现验证：从现有缓存重新计算主结果，与论文稿记录值比对（容差 ±0.02）。

覆盖（与 research/11_paper_methods_and_experiments.md 表 1/表 4 对应）：
  - BCSS 23 成员: val/test 的 mean / VOTE / wVOTE
  - BCSS 11×R38（同架构）: test mean / VOTE
  - LUAD 4×R38 变体: test mean / VOTE
  - LUAD 4×R38+2FM（跨架构 6 成员）: val/test mean / VOTE
  - LUAD 11×R38: test mean / VOTE

用法: python research/verify_repro.py   （产物: 控制台逐行, 与文档比对）
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from research.cacg import (discover, load_probs, labels_from, evaluate_gated,  # noqa
                           mean_cam, vote_ensemble)

BCSS_THR = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
LUAD_THR = np.full(4, 0.2, dtype=np.float32)

R11_BCSS = ["aug_s42", "bl_s42", "ch15_s42", "gsr_s42", "k15_s11", "k15_s21",
            "k15_s33", "k15_s42", "k15_s44", "k21_s42", "k7_s42"]
G4_LUAD = ["bl_s11", "ch15_s11", "gsr_s11", "k7_s11"]
FM_LUAD = ["pv1_seed11", "pv2s11"]
R11_LUAD = ["k15_s11", "k15_s42", "k15_s21", "k15_s33", "k15_s44", "bl_s11",
            "gsr_s11", "ch15_s11", "k7_s11", "k21_s11", "k15_s45"]

EXPECTED = {  # (split, pool) -> (mean, vote) 或 (mean, vote, wvote)
    ("bcss", "val", 23): (70.14, 70.43, 70.24),
    ("bcss", "test", 23): (71.99, 72.37, 72.40),
    ("bcss", "test", "R11"): (70.72, 70.76),
    ("luad", "val", "G4+FM2"): (77.07, 77.16),
    ("luad", "test", "G4+FM2"): (79.04, 78.57),
    ("luad", "test", "G4"): (78.13, 77.90),
    ("luad", "test", "R11"): (77.83, 77.54),
}


def run_pool(ds, split, tags):
    d = {x[0]: x for x in discover("/tmp/ens", ds, split)}
    mm = [d[t] for t in tags if t in d]
    P, g = load_probs([p for _, _, p in mm])
    thr = BCSS_THR if ds == "bcss" else LUAD_THR
    cam = mean_cam(mm, 256)
    m = evaluate_gated(cam, labels_from(P.mean(0), thr), g, 4)
    del cam
    v = vote_ensemble(mm, P, thr, g, 4, weighted=False)
    wv = vote_ensemble(mm, P, thr, g, 4, weighted=True)
    return dict(mean=m[0] * 100, vote=v[0] * 100, wvote=wv[0] * 100,
                mc=m[0] * 100, vc=v[0] * 100, cls_m=m[2] * 100, cls_v=v[2] * 100)


def main():
    rows = []
    pools = [("bcss", "val", 23, None), ("bcss", "test", 23, None),
             ("bcss", "test", "R11", R11_BCSS),
             ("luad", "val", "G4+FM2", G4_LUAD + FM_LUAD),
             ("luad", "test", "G4+FM2", G4_LUAD + FM_LUAD),
             ("luad", "test", "G4", G4_LUAD),
             ("luad", "test", "R11", R11_LUAD)]
    print(f"复现验证 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'数据集':6s} {'split':5s} {'池':10s} {'mean':>7s} {'VOTE':>7s} {'wVOTE':>7s}  期望(mean/VOTE)   结论", flush=True)
    ok = True
    for ds, split, pool, tags in pools:
        if tags is None:
            tags = [t for t, _, _ in discover("/tmp/ens", ds, split)]
        r = run_pool(ds, split, tags)
        e = EXPECTED[(ds, split, pool)]
        dv = abs(r["mean"] - e[0]); dvd = abs(r["vote"] - e[1])
        verdict = "PASS" if (dv <= 0.02 and dvd <= 0.02) else "FAIL"
        if verdict == "FAIL":
            ok = False
        rows.append((ds, split, pool, r, e, verdict))
        print(f"{ds:6s} {split:5s} {str(pool):10s} {r['mean']:7.2f} {r['vote']:7.2f} {r['wvote']:7.2f}  "
              f"{e[0]:.2f} / {e[1]:.2f}       {verdict}", flush=True)
    # BCSS test 逐类（mean vs VOTE）
    r = [x for x in rows if x[0] == "bcss" and x[1] == "test" and x[2] == 23][0]
    print("\nBCSS test 逐类 IoU（mean / VOTE）:", flush=True)
    print("  mean:", [f"{c:.1f}" for c in r[3]["cls_m"]], flush=True)
    print("  VOTE:", [f"{c:.1f}" for c in r[3]["cls_v"]], flush=True)
    print("\n== 总体:", "ALL PASS" if ok else "存在 FAIL ==")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()