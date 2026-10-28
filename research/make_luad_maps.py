# -*- coding: utf-8 -*-
"""为 LUAD test 生成 mean / VOTE / wVOTE / 单模型 预测并缓存（供定性图选样）。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "core_method"))

from core import (decision_voting, evaluate, labels_from,  # noqa: E402
                  member_decision, score_averaging)

CACHE = "/tmp/ens"
DS = "luad"
SPLIT = "test"
THR = np.full(4, 0.2, np.float32)
POOL = ["bl_s11", "ch15_s11", "gsr_s11", "k7_s11", "pv1_seed11", "pv2s11"]
SINGLE = "k15_s11"   # 单模型最优


def paths(tags):
    return [f"{CACHE}/{DS}_{t}_{SPLIT}_fused.npy" for t in tags]


def main():
    probs = np.stack([np.load(f"{CACHE}/{DS}_{t}_{SPLIT}_probs.npy").astype(np.float32)
                      for t in POOL])
    cam_paths = paths(POOL)

    sm = score_averaging(cam_paths, probs, THR)
    sv = decision_voting(cam_paths, probs, THR, 4)
    wv = decision_voting(cam_paths, probs, THR, 4, weighted=True)

    # 单模型预测（最优成员）
    p1 = np.load(f"{CACHE}/{DS}_{SINGLE}_{SPLIT}_probs.npy").astype(np.float32)
    A = np.load(f"{CACHE}/{DS}_{SINGLE}_{SPLIT}_fused.npy", mmap_mode="r")
    lab = labels_from(p1, THR).astype(np.float32)
    s1 = np.empty(A.shape[0], dtype=np.uint8)
    s1 = np.zeros(A.shape[:1] + A.shape[2:], dtype=np.uint8)
    for st in range(0, A.shape[0], 64):
        en = min(st + 64, A.shape[0])
        a = np.asarray(A[st:en], dtype=np.float32)
        s1[st:en] = (a * lab[st:en, :, None, None]).argmax(1).astype(np.uint8)

    out = os.path.join(ROOT, "research/figs/.maps_luad_test.npz")
    np.savez(out, Sm=sm.astype(np.uint8), Sv=sv.astype(np.uint8),
             wVOTE=wv.astype(np.uint8), single=s1)
    print("saved", out)
    gts = np.load(f"{CACHE}/{DS}_{SINGLE}_{SPLIT}_gts.npy", allow_pickle=True)
    for name, pred in [("single", s1), ("mean", sm), ("VOTE", sv), ("wVOTE", wv)]:
        print(f"  {name}: mIoU={evaluate(pred, list(gts), 4)[0]*100:.2f}")


if __name__ == "__main__":
    main()
