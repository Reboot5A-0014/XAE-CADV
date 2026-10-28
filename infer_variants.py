"""BCSS/LUAD 推理变体实验：量化官方 `tool/infer_fun.py` 之外的推理增强，并支持多模型集成。

官方默认做法：
  - TTA：仅 3 种（原图 / hflip / vflip）
  - CAM 融合：0.6*cam_28_1 + 0.2*cam_28_2 + 0.2*cam_deep（forward_cam 算出的 cam_56 未使用）
  - 图像级门控 label：仅用 deep 头 sigmoid 概率

本脚本一遍推理即可离线枚举：
  - TTA 集合：official(3) / d4(8，完整二面体群)
  - 门控来源：deep / mean4
  - CAM 融合权重：官方 (0,0.6,0.2,0.2) 及含 cam_56 的组合
  - 阈值：数据集默认 / 若干逐类组合
  - 多模型集成（--weights a.pth,b.pth）：概率与归一化 CAM 均按模型平均

用法：
  python infer_variants.py --weights ckptA.pth,ckptB.pth --tta-sets d4 \
      --dataset bcss --testroot datasets/BCSS-WSSS/test --n_class 4 \
      --img_size 224 --num_workers 8 --amp-dtype bf16
"""
import argparse
import importlib
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool.GenDataset import Stage1_InferDataset  # noqa: E402

cudnn.enabled = True

# (name, rot90 次数, 是否 hflip)；运算顺序 T = H^f ∘ R^k
ROT_OPS = [
    ("r0", 0, False), ("r90", 1, False), ("r180", 2, False), ("r270", 3, False),
    ("r0f", 0, True), ("r90f", 1, True), ("r180f", 2, True), ("r270f", 3, True),
]
TTA_SETS = {
    "official": ["r0", "r0f", "r180f"],   # 原图 / hflip / vflip
    "d4": [n for n, _, _ in ROT_OPS],     # 完整 8 元二面体群
}
OP_MAP = {n: (k, f) for n, k, f in ROT_OPS}


def apply_op(x, name):
    k, f = OP_MAP[name]
    if k:
        x = torch.rot90(x, k, dims=(2, 3))
    if f:
        x = torch.flip(x, dims=(3,))
    return x


def invert_cam(cam, name):
    k, f = OP_MAP[name]
    if f:
        cam = torch.flip(cam, dims=(2,))
    if k:
        cam = torch.rot90(cam, -k, dims=(1, 2))
    return cam


def norm_np(cam_np):
    c_min = np.min(cam_np, axis=(1, 2), keepdims=True)
    c_max = np.max(cam_np, axis=(1, 2), keepdims=True)
    return (cam_np - c_min) / (c_max - c_min + 1e-8)


def hist_iu_dice(hist):
    hist = hist.copy()
    hist[4, 4] = 0
    iu = np.diag(hist)[0:4] / ((hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))[0:4])
    dice = []
    for i in range(4):
        tp = np.diag(hist)[i]
        fp = hist[:, i].sum() - tp
        fn = hist[i, :].sum() - tp
        dice.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)
    return float(np.nanmean(iu)), float(np.mean(dice)), iu


@torch.no_grad()
def collect(model, dataroot, args, tta_sets):
    """单遍推理：对每个 TTA 集合缓存「聚合后的 4 个 CAM」与「4 个头概率」。"""
    net_cls = importlib.import_module(args.network)
    # 注意：resnet38d 覆写了 train() 且未 return self，nn.Module.eval() 会返回 None，不能链式调用
    model.eval()
    model = model.cuda()
    dataset = Stage1_InferDataset(data_path=os.path.join(dataroot, "img/"), img_size=args.img_size)
    loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp_dtype)

    cams_cache = {s: [] for s in tta_sets}
    probs_cache = {s: [] for s in tta_sets}
    gts = []
    for idx, (name_tuple, img_tensor) in enumerate(loader):
        img_name = name_tuple[0]
        orig_img = np.asarray(Image.open(os.path.join(dataroot, "img/", img_name + ".png")).convert("RGB"))
        orig_size = orig_img.shape[:2]
        x0 = img_tensor.cuda(non_blocking=True)

        for sname in tta_sets:
            acc_cams, acc_probs = None, None
            for op in TTA_SETS[sname]:
                x = apply_op(x0, op)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    # 一次前向得到 4 个 CAM 与 4 个头 logits（forward_cam 的信息超集）
                    out56, out281, out282, outdeep, y_deep, c56, c281, c282, cdeep, _ = \
                        net_cls.Net.forward(model, x)
                cams = [F.relu(c56), F.relu(c281), F.relu(c282), F.relu(cdeep)]
                cams = [F.interpolate(c, orig_size, mode="bilinear", align_corners=False)[0] for c in cams]
                cams = [invert_cam(c, op).detach().float().cpu().numpy() for c in cams]
                probs = torch.stack([torch.sigmoid(out56), torch.sigmoid(out281),
                                     torch.sigmoid(out282), y_deep]).detach().float().cpu().numpy()

                if acc_cams is None:
                    acc_cams, acc_probs = cams, probs
                else:
                    acc_cams = [a + b for a, b in zip(acc_cams, cams)]
                    acc_probs = acc_probs + probs
            n = len(TTA_SETS[sname])
            cams_cache[sname].append([a / n for a in acc_cams])
            probs_cache[sname].append((acc_probs / n)[:, 0, :])  # [4, C]

        gts.append(np.asarray(Image.open(os.path.join(dataroot, "mask/", img_name + ".png"))))
        if (idx + 1) % 500 == 0:
            print(f"  collected {idx + 1}/{len(dataset)}", flush=True)

    return cams_cache, probs_cache, gts


def evaluate(cams_list, probs_list, gts, n_class, gate, cam_w, thr_array, refine=None):
    """cams_list/probs_list 为长度 = 模型数的列表；集成时对归一化 CAM 与概率取平均。"""
    hist = np.zeros((n_class + 1, n_class + 1))
    n_models = len(cams_list)
    for i, gt in enumerate(gts):
        prob = np.mean([(p[i][3] if gate == "deep" else p[i].mean(axis=0)) for p in probs_list], axis=0)
        label = (prob > thr_array).astype(np.float32)
        if label.sum() == 0:
            label[int(np.argmax(prob))] = 1.0
        active = np.where(label > 1e-5)[0]

        cam = np.zeros_like(cams_list[0][i][0])
        for w, ci in zip(cam_w, range(4)):
            if w:
                cam += w * np.mean([norm_np(cm[i][ci]) for cm in cams_list], axis=0)
        cam = cam * label.reshape(n_class, 1, 1)
        if refine is not None and active.size:
            d, sc, ss = refine
            cam = np.stack([cv2.bilateralFilter(c.astype(np.float32), d, sc, ss) for c in cam])

        if active.size == 0:
            pred = np.full(gt.shape, n_class, dtype=np.int64)
        else:
            seg = np.asarray(cam[active].argmax(axis=0), dtype=np.int64)
            pred = np.zeros(gt.shape, dtype=np.int64)
            pred[:] = active[seg]
        gt_ = gt.astype(np.int64)
        pred = pred.copy()
        pred[gt_ == 4] = 4
        m = (gt_ >= 0) & (gt_ < n_class + 1)
        lab = (n_class + 1) * gt_[m] + pred[m]
        hist += np.bincount(lab, minlength=(n_class + 1) ** 2).reshape(n_class + 1, n_class + 1)
    return hist_iu_dice(hist)


def build_configs(dataset, n_models, tta_sets):
    if dataset == "bcss":
        d_thr = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
        thr_variants = {
            "thr默认": d_thr,
            "thr-0.1": np.asarray([0.7, 0.8, 0.7, 0.5], dtype=np.float32),
            "thr+0.05": np.asarray([0.85, 0.95, 0.85, 0.65], dtype=np.float32),
        }
    else:
        d_thr = np.full(4, 0.2, dtype=np.float32)
        thr_variants = {"thr默认": d_thr, "thr0.3": np.full(4, 0.3, dtype=np.float32)}

    weights = {
        "官方(0/0.6/0.2/0.2)": (0.0, 0.6, 0.2, 0.2),
        "含56:0.15/0.5/0.15/0.2": (0.15, 0.5, 0.15, 0.2),
        "含56:0.2/0.5/0.15/0.15": (0.2, 0.5, 0.15, 0.15),
        "均衡(0.25/0.25/0.25/0.25)": (0.25, 0.25, 0.25, 0.25),
    }
    gates = ("deep", "mean4") if n_models == 1 else ("deep", "mean4")

    refine_variants = {
        "raw": None,
        "bf5/0.05/5": (5, 0.05, 5),
        "bf7/0.10/7": (7, 0.10, 7),
        "bf9/0.20/9": (9, 0.20, 9),
        "bf11/0.30/11": (11, 0.30, 11),
    }
    configs = []
    for sname in tta_sets:
        for gate in gates:
            for wname, w in weights.items():
                for tname, thr in thr_variants.items():
                    for rname, refine in refine_variants.items():
                        configs.append((f"m{n_models}|{sname}|{gate}|{wname}|{tname}|{rname}",
                                        sname, gate, w, thr, refine))
    return configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="network.resnet38_cls")
    p.add_argument("--dataset", default="bcss")
    p.add_argument("--testroot", default="datasets/BCSS-WSSS/val")
    p.add_argument("--weights", required=True, help="逗号分隔，可传 1~2 个检查点做集成")
    p.add_argument("--tta-sets", default="official,d4")
    p.add_argument("--n_class", default=4, type=int)
    p.add_argument("--img_size", default=224, type=int)
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--amp_dtype", "--amp-dtype", default="bf16", choices=["none", "bf16", "fp16"])
    args = p.parse_args()

    tta_sets = args.tta_sets.split(",")
    ckpts = args.weights.split(",")

    cams_all, probs_all, gts = [], [], None
    for ck in ckpts:
        model = getattr(importlib.import_module(args.network), "Net_CAM")(n_class=args.n_class)
        model.load_state_dict(torch.load(ck), strict=False)
        print(f"[collect] {ck} | {args.testroot} | TTA={tta_sets}", flush=True)
        cams, probs, gts = collect(model, args.testroot, args, tta_sets)
        cams_all.append(cams)
        probs_all.append(probs)
        del model
        torch.cuda.empty_cache()

    results = []
    for name, sname, gate, w, thr, refine in build_configs(args.dataset, len(ckpts), tta_sets):
        cams_list = [c[sname] for c in cams_all]
        probs_list = [pr[sname] for pr in probs_all]
        miou, mdice, iu = evaluate(cams_list, probs_list, gts, args.n_class, gate, w, thr, refine)
        results.append((miou, mdice, name, iu))
        print(f"{name:58s} mIoU={miou*100:6.2f} mDice={mdice*100:6.2f} "
              f"cls=[{iu[0]*100:.1f} {iu[1]*100:.1f} {iu[2]*100:.1f} {iu[3]*100:.1f}]", flush=True)

    print("\n=== TOP 15 (by mIoU) ===")
    for miou, mdice, name, iu in sorted(results, reverse=True)[:15]:
        print(f"{miou*100:6.2f} / {mdice*100:6.2f}  {name}")


if __name__ == "__main__":
    main()
