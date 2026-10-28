"""为 CACG 模块采集成员缓存（融合 CAM + 图像级概率），存为可 mmap 的 .npy。

用法：
  python research/cacg_collect.py --dataset bcss --family r38
  PHIKON_ID=owkin/phikon    python research/cacg_collect.py --dataset bcss --family pv1
  PHIKON_ID=owkin/phikon-v2 python research/cacg_collect.py --dataset bcss --family pv2
"""
import argparse
import importlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.softgate_sweep import collect  # noqa: E402

# (tag, weights, gsr, ch, kernel)
R38 = {
    "bcss": [
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
    ],
}

PV1 = ["seed42", "s11", "s2024", "e100", "ft4", "ft4lr", "448"]
# (tag, 权重目录名)，tag 与 09 文档 / ensemble 缓存命名一致
PV2 = [("pv2", "pv2"), ("pv2s11", "pv2_s11"), ("pv2s21", "pv2_s21"),
       ("pv2s33", "pv2_s33"), ("pv2s44", "pv2_s44")]


def phikon_specs(family, dataset="bcss"):
    if dataset == "luad":
        # LUAD 上可用的基础模型成员（与 checkpoints_phikon/checkpoints_p2fm 实际权重对应）
        if family == "pv1":
            return [("pv1_seed11", "checkpoints_phikon/luad_seed11/stage1_last.pth"),
                    ("pv1_ft4", "checkpoints_phikon/luad_ft4/stage1_last.pth"),
                    ("pv1_ft4lr", "checkpoints_phikon/luad_ft4lr/stage1_last.pth")]
        return [("pv2s11", "checkpoints_p2fm/luad_pv2_s11/stage1_last.pth")]
    if family == "pv1":
        return [(f"pv1_{t}", f"checkpoints_phikon/bcss_{t}/stage1_last.pth") for t in PV1]
    return [(tag, f"checkpoints_p2fm/bcss_{d}/stage1_last.pth") for tag, d in PV2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bcss")
    ap.add_argument("--family", default="r38",
                    choices=["r38", "pv1", "pv2", "cnx", "swin", "r50", "cnxft", "swinft", "r50ft"])
    ap.add_argument("--cache_dir", default="/tmp/ens")
    ap.add_argument("--n_class", default=4, type=int)
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--num_workers", default=8, type=int)
    ap.add_argument("--amp_dtype", "--amp-dtype", default="bf16")
    ap.add_argument("--cam_w", default="0.6,0.2,0.2")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--tags", default="", help="逗号分隔，仅采集列出的 tag（用于并行分片）")
    args = ap.parse_args()
    args.weights_tuple = tuple(float(x) for x in args.cam_w.split(","))
    os.makedirs(args.cache_dir, exist_ok=True)

    splits = args.splits.split(",")
    root = os.path.join("datasets", "BCSS-WSSS" if args.dataset == "bcss" else "LUAD-HistoSeg")

    if args.family == "r38":
        specs = [(t, w, {"hfrm_gsr": g, "hfrm_ch": c, "hfrm_kernel": k})
                 for (t, w, g, c, k) in R38[args.dataset]]
        net_mod = "network.resnet38_cls"
    elif args.family in ("cnx", "swin", "r50", "cnxft", "swinft", "r50ft"):
        base = args.family.replace("ft", "")
        d, tag = {"cnx": ("convnext", "t_cnx"), "swin": ("swin", "t_swin"),
                  "r50": ("resnet50", "t_r50")}[base]
        ckpt_root = "checkpoints_timm_ft" if args.family.endswith("ft") else "checkpoints_timm"
        tag = tag + ("_ft" if args.family.endswith("ft") else "")
        specs = [(tag, f"{ckpt_root}/luad_{d}/stage1_last.pth", {})]
        net_mod = "network.timm_cam"
    else:
        specs = [(t, w, {}) for t, w in phikon_specs(args.family, args.dataset)]
        net_mod = "network.phikon_cam"

    if args.tags:
        want = set(x for x in args.tags.split(",") if x)
        specs = [s for s in specs if s[0] in want]

    for tag, w, kw in specs:
        if not os.path.exists(w):
            print(f"[skip] {tag}: 缺权重 {w}", flush=True)
            continue
        f_fused = f"{args.cache_dir}/{args.dataset}_{tag}_{splits[0]}_fused.npy"
        done = all(os.path.exists(f"{args.cache_dir}/{args.dataset}_{tag}_{s}_fused.npy")
                   and os.path.exists(f"{args.cache_dir}/{args.dataset}_{tag}_{s}_probs.npy")
                   for s in splits)
        if done:
            print(f"[cached] {tag}", flush=True)
            continue
        model = getattr(importlib.import_module(net_mod), "Net_CAM")(n_class=args.n_class, **kw)
        model.load_state_dict(torch.load(w, map_location="cpu"), strict=False)
        for split in splits:
            f_fused = f"{args.cache_dir}/{args.dataset}_{tag}_{split}_fused.npy"
            f_probs = f"{args.cache_dir}/{args.dataset}_{tag}_{split}_probs.npy"
            f_gts = f"{args.cache_dir}/{args.dataset}_{tag}_{split}_gts.npy"
            if os.path.exists(f_fused) and os.path.exists(f_probs):
                print(f"[cached] {tag} {split}", flush=True)
                continue
            fused, probs, gts = collect(model, os.path.join(root, split),
                                        args, args.n_class, args.weights_tuple)
            np.save(f_fused, fused)
            np.save(f_probs, probs.astype(np.float32))
            np.save(f_gts, np.array(gts, dtype=object), allow_pickle=True)
            print(f"[saved] {tag} {split} fused={fused.shape} probs={probs.shape}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
