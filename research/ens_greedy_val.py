"""在 VAL 上做贪心成员剔除（后向消除），再在 TEST 上报数。

适用于 val 规模较小的数据集（LUAD val=300 张），可把全部成员的 val 缓存常驻内存。
BCSS val=3418 张时内存过大，请改用 ens_weight_select.py 的族间配比方案。

用法：python research/ens_greedy_val.py --dataset luad
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.ensemble_ens import evaluate  # noqa: E402


def load(d, ds, split, tags, val_subset=0):
    out = {}
    gts = None
    for t in tags:
        f = f"{d}/{ds}_{t}_{split}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        if val_subset and split == "val":
            out[t] = (z["probs"][:val_subset], z["fused"][:val_subset])
        else:
            out[t] = (z["probs"], z["fused"])
        gts = list(z["gts"])[:val_subset] if (val_subset and split == "val") else list(z["gts"])
        del z
    return out, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="luad")
    ap.add_argument("--dir", default="/tmp/ens")
    ap.add_argument("--n_class", default=4, type=int)
    ap.add_argument("--min_keep", default=4, type=int)
    ap.add_argument("--val_subset", default=0, type=int, help="只用前 N 张 val 做选择（控制内存）")
    args = ap.parse_args()

    thr = (np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32) if args.dataset == "bcss"
           else np.full(args.n_class, 0.2, dtype=np.float32))

    def tags_of(split):
        return sorted(os.path.basename(f)[len(args.dataset) + 1:-len(f"_{split}.npz")]
                      for f in glob.glob(f"{args.dir}/{args.dataset}_*_{split}.npz"))

    tags = sorted(set(tags_of("val")) & set(tags_of("test")))
    print(f"候选成员（{len(tags)}）：{tags}\n")
    V, gv = load(args.dir, args.dataset, "val", tags, args.val_subset)

    def ens_score(keys, split_data, gts):
        ps = sum(split_data[k][0] for k in keys) / len(keys)
        cs = sum(split_data[k][1].astype(np.float32) for k in keys) / len(keys)
        return evaluate(ps, cs, gts, thr, args.n_class)[0] * 100

    cur = [t for t in tags if t in V]
    print("各成员 val 单模型 mIoU：")
    for t in cur:
        print(f"  {t:12s} {ens_score([t], V, gv):6.2f}")

    best = ens_score(cur, V, gv)
    print(f"\n初始（{len(cur)} 个）val = {best:.2f}")
    hist = []
    while len(cur) > args.min_keep:
        cand = None
        for t in cur:
            k = [x for x in cur if x != t]
            s = ens_score(k, V, gv)
            if cand is None or s > cand[1]:
                cand = (t, s)
        if cand[1] <= best + 1e-6:
            break
        print(f"  剔除 {cand[0]:12s} → val {cand[1]:.2f} (+{cand[1]-best:.2f})")
        hist.append(cur.pop(cur.index(cand[0])))
        best = cand[1]
    print(f"\nval 选出的成员（{len(cur)}）：{cur}\n")

    T, gt = load(args.dir, args.dataset, "test", tags)
    ref = 71.82 if args.dataset == "bcss" else 77.93
    for name, keys in [("全部成员", tags), ("val 贪心剔除后", cur)]:
        m, dv, iu = evaluate(sum(T[k][0] for k in keys) / len(keys),
                             sum(T[k][1].astype(np.float32) for k in keys) / len(keys),
                             gt, thr, args.n_class)
        print(f"[TEST] {name}（{len(keys)}）: mIoU = {m*100:.2f}  mDice = {dv*100:.2f}  "
              f"{'✅ 超过' if m*100 > ref else '❌ 未超过'} 论文 {ref}")
    print(f"\n文献值：BCSS 71.82±0.38 | LUAD 77.93±0.24")


if __name__ == "__main__":
    main()
