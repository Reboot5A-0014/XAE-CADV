"""跨类标定（soft calibration）离线扫描。

假设：官方推理 `tool/infer_fun.py` 对每个类通道做**逐类 min-max 归一化**（max 恒为 1），
再用 0/1 门控 `cam = cam * label` 二值相乘。这导致两个问题：
  1. 假阳类（过了阈值但实际不存在）通道被抬到 max=1，在 argmax 中抢走正确类的像素；
  2. 类间幅度不可比 —— "证据弱"与"证据强"的类在竞争时地位相同。

本脚本缓存**融合后的 CAM**（与官方完全一致：各尺度逐类 min-max 后按权重加权求和）
与图像级概率，然后离线评估一族"跨类标定"函数 c_k(prob) 替换官方 0/1 门控：
    cam_k  <-  c_k * fused_cam_k
若某一族标定显著优于官方硬门控，则说明该假设成立，可作为方法论文的核心模块。

用法：
  python research/softgate_sweep.py --weights checkpoints/bcss_seed42/stage1_last.pth \
      --dataset bcss --testroot datasets/BCSS-WSSS/val --n_class 4 --img_size 224
"""
import argparse
import importlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool.GenDataset import Stage1_InferDataset  # noqa: E402

cudnn.enabled = True

TTA_TRANSFORMS = (((), ()), ((3,), (2,)), ((2,), (1,)))


def norm_np(cam_np):
    """与 infer_fun.norm_np 完全一致：逐类 min-max。"""
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


@torch.no_grad()
def collect(model, dataroot, args, n_class, weights):
    """单遍推理，缓存融合 CAM（fp16 预分配，避免 OOM）与图像级概率。"""
    model.eval()
    model = model.cuda()
    dataset = Stage1_InferDataset(data_path=os.path.join(dataroot, "img/"), img_size=args.img_size)
    loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    amp_dtype = None
    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "fp16":
        amp_dtype = torch.float16

    n = len(dataset)
    shape = None
    fused = None       # [N, C, H, W] fp16
    probs = np.zeros((n, n_class), dtype=np.float32)
    gts = []

    w1, w2, w3 = weights
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

        # 与 infer_fun 一致：各尺度先逐类 min-max，再按权重融合
        cam = w1 * norm_np(c1) + w2 * norm_np(c2) + w3 * norm_np(cd)

        if shape is None:
            shape = cam.shape
            fused = np.zeros((n,) + shape, dtype=np.float16)
        fused[idx] = cam.astype(np.float16)
        probs[idx] = prob
        gts.append(np.asarray(Image.open(os.path.join(dataroot, "mask/", img_name + ".png"))))

        if (idx + 1) % 500 == 0:
            print(f"  collected {idx + 1}/{n}", flush=True)

    return fused, probs, gts


def calibrate_weights(probs, mode, thr, gamma):
    """返回 [N, C] 的跨类标定权重 c_k。"""
    s = probs
    if mode == "hard":                       # 官方：0/1 硬门控
        w = (s > thr).astype(np.float32)
        empty = w.sum(axis=1) == 0
        if empty.any():
            w[empty, np.argmax(s[empty], axis=1)] = 1.0
        return w
    if mode == "pow":                        # c_k = s_k^gamma（无硬门）
        return np.power(np.clip(s, 1e-6, 1.0), gamma).astype(np.float32)
    if mode == "ramp":                       # c_k = clip((s_k - thr)/(1-thr), 0, 1)
        return np.clip((s - thr) / (1.0 - thr), 0.0, 1.0).astype(np.float32)
    if mode == "hard_pow":                   # 过阈值的类按 s_k^gamma 加权，未过仍为 0
        w = calibrate_weights(s, "hard", thr, gamma)
        return (w * np.power(np.clip(s, 1e-6, 1.0), gamma)).astype(np.float32)
    if mode == "hard_ramp":                  # 过阈值的类按 ramp 加权（等价 hard，仅示意）
        w = calibrate_weights(s, "hard", thr, gamma)
        return (w * np.clip((s - thr) / (1.0 - thr + 1e-8), 0.0, 1.0)).astype(np.float32)
    if mode == "softmax":                    # 类间竞争归一化：C * softmax(s / gamma)
        z = s / max(gamma, 1e-6)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(axis=1, keepdims=True) * s.shape[1]).astype(np.float32)
    if mode == "top1":                       # 只保留最可能的一类（精度极端偏置）
        w = np.zeros_like(s)
        w[np.arange(len(s)), np.argmax(s, axis=1)] = 1.0
        return w
    raise ValueError(mode)


def evaluate(fused, probs, gts, n_class, mode, thr, gamma):
    hist = np.zeros((n_class + 1, n_class + 1))
    w_all = calibrate_weights(probs, mode, thr, gamma)
    for i, gt in enumerate(gts):
        w = w_all[i]
        cam = fused[i].astype(np.float32) * w.reshape(n_class, 1, 1)
        pred = np.asarray(cam.argmax(axis=0), dtype=np.int64)
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
    parser.add_argument("--amp_dtype", "--amp-dtype", default="bf16",
                        choices=["none", "bf16", "fp16"], type=str)
    parser.add_argument("--cam_w", default="0.6,0.2,0.2", type=str)
    parser.add_argument("--cache", default=None, type=str, help="缓存 npz 路径（复用推理结果）")
    parser.add_argument("--collect_only", action="store_true", help="只采集并保存缓存，不做标定扫描")
    args = parser.parse_args()

    if args.dataset == "bcss":
        default_thr = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
    else:
        default_thr = np.full(4, 0.2, dtype=np.float32)
    weights = tuple(float(x) for x in args.cam_w.split(","))

    if args.cache and os.path.exists(args.cache):
        print(f"[cache] load {args.cache}", flush=True)
        z = np.load(args.cache, allow_pickle=True)
        fused, probs = z["fused"], z["probs"]
        gts = list(z["gts"])
    else:
        model = getattr(importlib.import_module(args.network), "Net_CAM")(n_class=args.n_class)
        model.load_state_dict(torch.load(args.weights), strict=False)
        print(f"[collect] dataset={args.dataset} root={args.testroot} w={weights}", flush=True)
        fused, probs, gts = collect(model, args.testroot, args, args.n_class, weights)
        if args.cache:
            np.savez_compressed(args.cache, fused=fused, probs=probs,
                                gts=np.array(gts, dtype=object))
            print(f"[cache] saved {args.cache}", flush=True)

    if args.collect_only:
        print("[collect_only] 已保存缓存，退出", flush=True)
        return

    results = []

    def run(name, mode, thr, gamma):
        miou, mdice, iu = evaluate(fused, probs, gts, args.n_class, mode, thr, gamma)
        results.append((miou, mdice, name, iu))
        print(f"{name:46s} mIoU={miou*100:6.2f}  mDice={mdice*100:6.2f}  "
              f"clsIoU=[{iu[0]*100:.1f} {iu[1]*100:.1f} {iu[2]*100:.1f} {iu[3]*100:.1f}]",
              flush=True)
        return miou

    base = run("官方 硬门控 thr=default", "hard", default_thr, 1.0)

    print("\n--- 族 A: 无硬门，c_k = s_k^gamma ---", flush=True)
    for g in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        run(f"pow  gamma={g}", "pow", default_thr, g)

    print("\n--- 族 B: 无硬门，c_k = clip((s_k-thr)/(1-thr),0,1) ---", flush=True)
    for t in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7):
        run(f"ramp thr={t}", "ramp", np.full(4, t, dtype=np.float32), 1.0)

    print("\n--- 族 C: 硬门 + 类内再加权 c_k = 1[s>thr] * s_k^gamma ---", flush=True)
    for g in (0.5, 1.0, 2.0, 3.0):
        run(f"hard_pow gamma={g}", "hard_pow", default_thr, g)

    print("\n--- 族 D: 类间竞争归一 c_k = C*softmax(s/tau) ---", flush=True)
    for tau in (0.05, 0.1, 0.2, 0.5):
        run(f"softmax tau={tau}", "softmax", default_thr, tau)

    print("\n--- 族 E: 只留 top-1 类 ---", flush=True)
    run("top1", "top1", default_thr, 1.0)

    print("\n=== TOP 12 (by mIoU) ===")
    for miou, mdice, name, iu in sorted(results, reverse=True)[:12]:
        delta = (miou - base) * 100
        print(f"{miou*100:6.2f} / {mdice*100:6.2f}  ({delta:+.2f})  {name}")


if __name__ == "__main__":
    main()
