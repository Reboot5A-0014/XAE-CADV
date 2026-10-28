"""BCSS/LUAD CAM 推理超参扫描（单遍推理 + 离线多配置评估）。

目的：定位推理阶段 `tool/infer_fun.py` 里两个数据集相关超参对结果的影响：
  1. 图像级阈值 `class_thresholds`（bcss 默认 [0.8, 0.9, 0.8, 0.6]；luad 默认 0.2）
  2. CAM 融合权重 (cam_w_28_1, cam_w_28_2, cam_w_deep)（默认 0.6/0.2/0.2）

做法：对数据集的每张图只做一次 3 路 TTA 前向，缓存三路归一化后的 CAM 与 sigmoid 概率，
         然后对所有候选配置离线计算 seg_map 与混淆矩阵，一次扫描给出全部结果。

用法：
  python sweep_infer.py --weights checkpoints_bcss_seed11/stage1_last.pth \
      --dataset bcss --testroot datasets/BCSS-WSSS/val --n_class 4 \
      --img_size 224 --num_workers 8 --amp-dtype bf16
"""
import argparse
import importlib
import os
import sys
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tool.GenDataset import Stage1_InferDataset  # noqa: E402

cudnn.enabled = True

# 与 infer_fun._tta_transforms() 保持一致
TTA_TRANSFORMS = (((), ()), ((3,), (2,)), ((2,), (1,)))


def norm_np(cam_np):
    c_min = np.min(cam_np, axis=(1, 2), keepdims=True)
    c_max = np.max(cam_np, axis=(1, 2), keepdims=True)
    return (cam_np - c_min) / (c_max - c_min + 1e-8)


def hist_iu_dice(hist):
    """与 tool/iouutils.scores 完全一致的口径（仅统计前 4 个前景类）。"""
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


def build_configs(dataset):
    """返回 [(name, thresholds(np.ndarray[4] 或 None 表示 oracle 用 GT 标签), (w1, w2, w3)), ...]"""
    if dataset == "bcss":
        default_thr = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
    else:
        default_thr = np.full(4, 0.2, dtype=np.float32)

    scalar_thrs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    weight_sets = {
        "w0.6/0.2/0.2(默认)": (0.6, 0.2, 0.2),
        "w0.11/0.78/0.11(参考配置)": (0.11, 0.78, 0.11),
        "w0.2/0.6/0.2": (0.2, 0.6, 0.2),
        "w0.34/0.33/0.33": (0.34, 0.33, 0.33),
        "w0.8/0.1/0.1": (0.8, 0.1, 0.1),
    }

    configs = []
    # oracle：图像级标签直接用 GT（mask 反推），用于估计当前 CAM 的精度上界
    configs.append(("ORACLE_gt_label|w0.6/0.2/0.2", None, (0.6, 0.2, 0.2)))
    configs.append(("ORACLE_gt_label|w0.34/0.33/0.33", None, (0.34, 0.33, 0.33)))
    for wname, w in weight_sets.items():
        configs.append((f"default_thr|{wname}", default_thr, w))
    for thr, (wname, w) in product(scalar_thrs, weight_sets.items()):
        configs.append((f"thr{thr}|{wname}", np.full(4, thr, dtype=np.float32), w))
    # 逐类阈值坐标搜索（其余类保持默认阈值）
    for c in range(4):
        for thr in scalar_thrs:
            t = default_thr.copy()
            t[c] = thr
            configs.append((f"percls_c{c}={thr}|w默认", t, (0.6, 0.2, 0.2)))
    return configs



@torch.no_grad()
def collect(model, dataroot, args, n_class):
    """单遍推理：返回每张图的三路归一化 CAM、概率、GT mask。"""
    # 注意：network/resnet38d.py 覆写了 train() 且未 return self，
    # 导致 nn.Module.eval()（实现为 return self.train(False)）返回 None，不能链式调用。
    model.eval()
    model = model.cuda()
    dataset = Stage1_InferDataset(data_path=os.path.join(dataroot, "img/"), img_size=args.img_size)
    loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    amp_dtype = None
    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "fp16":
        amp_dtype = torch.float16

    cams, probs, gts, names = [], [], [], []
    for idx, (img_name_tuple, img_tensor) in enumerate(loader):
        img_name = img_name_tuple[0]
        img_path = os.path.join(dataroot, "img/", img_name + ".png")
        orig_img = np.asarray(Image.open(img_path).convert("RGB"))
        orig_size = orig_img.shape[:2]
        img_tensor = img_tensor.cuda(non_blocking=True)

        acc_1, acc_2, acc_d, acc_y = [], [], [], []
        for in_flip, cam_flip in TTA_TRANSFORMS:
            tta = torch.flip(img_tensor, dims=in_flip) if in_flip else img_tensor
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                _, c1, c2, cd, y = model.forward_cam(tta)
                c1 = F.interpolate(c1, orig_size, mode="bilinear", align_corners=False)[0]
                c2 = F.interpolate(c2, orig_size, mode="bilinear", align_corners=False)[0]
                cd = F.interpolate(cd, orig_size, mode="bilinear", align_corners=False)[0]
            if cam_flip:
                c1, c2, cd = (torch.flip(t, dims=cam_flip) for t in (c1, c2, cd))
            acc_1.append(c1)
            acc_2.append(c2)
            acc_d.append(cd)
            acc_y.append(y)

        c1 = torch.stack(acc_1).mean(0).detach().float().cpu().numpy()
        c2 = torch.stack(acc_2).mean(0).detach().float().cpu().numpy()
        cd = torch.stack(acc_d).mean(0).detach().float().cpu().numpy()
        prob = torch.stack(acc_y).mean(0).detach().float().cpu().numpy()[0]

        cams.append((norm_np(c1), norm_np(c2), norm_np(cd)))
        probs.append(prob)
        gts.append(np.asarray(Image.open(os.path.join(dataroot, "mask/", img_name + ".png"))))
        names.append(img_name)
        if (idx + 1) % 500 == 0:
            print(f"  collected {idx + 1}/{len(dataset)}", flush=True)

    return cams, np.stack(probs), gts, names


def evaluate(cams, probs, gts, n_class, thresholds, weights):
    hist = np.zeros((n_class + 1, n_class + 1))
    w1, w2, w3 = weights
    for (n1, n2, n3), prob, gt in zip(cams, probs, gts):
        if thresholds is None:
            # oracle：直接用 mask 反推的图像级标签做门控
            label = np.asarray([1.0 if (gt == c).any() else 0.0 for c in range(n_class)], dtype=np.float32)
        else:
            label = (prob > thresholds).astype(np.float32)
            if label.sum() == 0:
                label[int(np.argmax(prob))] = 1.0
        active = np.where(label > 1e-5)[0]
        cam = w1 * n1 + w2 * n2 + w3 * n3
        cam = cam * label.reshape(n_class, 1, 1)
        if active.size == 0:  # 整图无前景（oracle 时可能出现）
            seg_full = np.full(gt.shape, n_class, dtype=np.int64)
        else:
            seg = np.asarray(cam[active].argmax(axis=0), dtype=np.int64)
            seg_full = np.zeros(gt.shape, dtype=np.int64)
            # argmax 在 active 子集上做，等价于在补零后的全部通道上做 argmax
            seg_full[:] = active[seg]
        pred = seg_full
        gt_ = gt.astype(np.int64)
        pred = pred.copy()
        pred[gt_ == 4] = 4
        mask = (gt_ >= 0) & (gt_ < n_class + 1)
        lab = (n_class + 1) * gt_[mask] + pred[mask]
        hist += np.bincount(lab, minlength=(n_class + 1) ** 2).reshape(n_class + 1, n_class + 1)
    return hist_iu_dice(hist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--dataset", default="bcss", type=str)
    parser.add_argument("--testroot", default="datasets/BCSS-WSSS/val", type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--amp_dtype", "--amp-dtype", default="bf16", choices=["none", "bf16", "fp16"], type=str)
    parser.add_argument("--dump", default=None, type=str)
    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), "Net_CAM")(n_class=args.n_class)
    model.load_state_dict(torch.load(args.weights), strict=False)

    print(f"[collect] dataset={args.dataset} root={args.testroot}", flush=True)
    cams, probs, gts, names = collect(model, args.testroot, args, args.n_class)
    if args.dump:
        gt_presence = np.stack([[(g == c).any() for c in range(args.n_class)] for g in gts]).astype(np.float32)
        np.savez_compressed(args.dump, probs=probs, gt_presence=gt_presence, names=np.array(names))
        print(f"[dump] saved {args.dump}", flush=True)
        return

    results = []
    for name, thr, w in build_configs(args.dataset):
        miou, mdice, iu = evaluate(cams, probs, gts, args.n_class, thr, w)
        results.append((miou, mdice, name, iu, thr, w))
        print(f"{name:40s} mIoU={miou*100:6.2f}  mDice={mdice*100:6.2f}  "
              f"clsIoU=[{iu[0]*100:.1f} {iu[1]*100:.1f} {iu[2]*100:.1f} {iu[3]*100:.1f}]", flush=True)

    print("\n=== TOP 15 (by mIoU) ===")
    for miou, mdice, name, iu, thr, w in sorted(results, reverse=True)[:15]:
        thr_txt = "ORACLE" if thr is None else np.round(thr, 2).tolist()
        print(f"{miou*100:6.2f} / {mdice*100:6.2f}  {name}  thr={thr_txt} w={w}")


if __name__ == "__main__":
    main()
