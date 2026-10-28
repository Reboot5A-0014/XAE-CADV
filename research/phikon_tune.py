"""为 Phikon 骨干模型重新标定推理超参（逐类阈值 + 多尺度融合权重 + 门控来源）。

动机：官方阈值 [0.8,0.9,0.8,0.6] 与融合权重 0.6/0.2/0.2 是为 ResNet38 调的；
换骨干后 CAM 分布改变，超参失配（Phikon val +2.80 但 test 持平即其症状）。

用法：
  python research/phikon_tune.py --weights checkpoints_phikon/bcss_seed42/stage1_last.pth \
      --dataset bcss --valroot datasets/BCSS-WSSS/val --testroot datasets/BCSS-WSSS/test
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool.GenDataset import Stage1_InferDataset      # noqa: E402
from tool import iouutils                            # noqa: E402

TTA = (((), ()), ((3,), (2,)), ((2,), (1,)))


def norm_np(a):
    mn = a.min(axis=(1, 2), keepdims=True); mx = a.max(axis=(1, 2), keepdims=True)
    return (a - mn) / (mx - mn + 1e-8)


@torch.no_grad()
def collect(weights_path, network, root, n_class, img_size, num_workers):
    import importlib
    net_cls = importlib.import_module(network)
    model = net_cls.Net_CAM(n_class=n_class)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
    model.eval(); model = model.cuda()
    ds = Stage1_InferDataset(data_path=os.path.join(root, "img/"), img_size=img_size)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=num_workers)
    # 预分配 float16 存储，避免 np.stack 复制大数组触发 32GB cgroup OOM
    # S: [N, 4 尺度, C 类别, H, W]
    S = np.zeros((len(ds), 4, n_class, img_size, img_size), dtype=np.float16)
    P = np.zeros((len(ds), n_class), dtype=np.float32)
    gts = [None] * len(ds)
    idx = 0
    for names, img in dl:
        x0 = img.cuda(non_blocking=True)
        cs, ps = None, []
        for d_in, d_cam in TTA:
            x = torch.flip(x0, dims=d_in) if d_in else x0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c56, c1, c2, cd, y = model.forward_cam(x)
            outs = [F.interpolate(c.float(), size=(img_size, img_size), mode="bilinear",
                                  align_corners=False) for c in (c56, c1, c2, cd)]
            if d_cam:
                off = tuple(d + 1 for d in d_cam)
                outs = [torch.flip(o, dims=off) for o in outs]
            cs = outs if cs is None else [a + b for a, b in zip(cs, outs)]
            ps.append(y.float())
        pm = torch.stack(ps).mean(0).float().cpu().numpy()
        for b, nm in enumerate(names):
            arr = [norm_np(c[b].float().cpu().numpy()) for c in cs]
            for k in range(4):
                S[idx, k] = arr[k]
            P[idx] = pm[b]
            gts[idx] = np.asarray(Image.open(os.path.join(root, "mask", nm + ".png")))
            idx += 1
        if idx % 800 == 0:
            print("      collected %d/%d" % (idx, len(ds)), flush=True)
    del model; torch.cuda.empty_cache()
    return S, P, gts


def evaluate(fused, prob, gts, thr, n_class):
    preds = []
    for i in range(len(gts)):
        lab = (prob[i] > thr).astype(np.float32)
        if lab.sum() == 0:
            lab[int(np.argmax(prob[i]))] = 1.0
        preds.append(np.argmax(fused[i] * lab.reshape(-1, 1, 1), axis=0).astype(int))
    return iouutils.scores(gts, preds, n_class=n_class)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--network", default="network.phikon_cam")
    ap.add_argument("--dataset", default="bcss")
    ap.add_argument("--valroot", default="datasets/BCSS-WSSS/val")
    ap.add_argument("--testroot", default="datasets/BCSS-WSSS/test")
    ap.add_argument("--n_class", default=4, type=int)
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--num_workers", default=8, type=int)
    args = ap.parse_args()
    nC = args.n_class

    print("[1/4] 收集 val", flush=True); t0 = time.time()
    Sva, Pva, Gva = collect(args.weights, args.network, args.valroot, nC, args.img_size, args.num_workers)
    print("      val 收集完成 %.0fs  scales=%s" % (time.time() - t0, tuple(Sva.shape)), flush=True)

    def fuse(S, w):
        f = np.zeros((S.shape[0], S.shape[2], S.shape[3], S.shape[4]), dtype=np.float32)
        for k in range(4):
            if w[k]:
                f += w[k] * S[:, k].astype(np.float32)
        return f

    weights = {
        "官方(0/0.6/0.2/0.2)": (0.0, 0.6, 0.2, 0.2),
        "含56(0.2/0.5/0.2/0.1)": (0.2, 0.5, 0.2, 0.1),
        "含56(0.1/0.5/0.2/0.2)": (0.1, 0.5, 0.2, 0.2),
        "均衡(0.25x4)": (0.25, 0.25, 0.25, 0.25),
        "重深层(0/0.3/0.2/0.5)": (0.0, 0.3, 0.2, 0.5),
        "深层为主(0/0.2/0.2/0.6)": (0.0, 0.2, 0.2, 0.6),
    }
    d_thr = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32) if args.dataset == "bcss" \
        else np.full(nC, 0.2, dtype=np.float32)

    print("\n[2/4] 融合权重搜索（默认阈值）", flush=True)
    best_w, best_m = None, -1
    for name, w in weights.items():
        f = fuse(Sva, w)
        s = evaluate(f, Pva, Gva, d_thr, nC)["Mean IoU"]
        print("  %-24s val mIoU %.4f" % (name, s), flush=True)
        if s > best_m:
            best_m, best_w, best_name = s, w, name
    print("  -> 最佳权重 %s %.4f" % (best_name, best_m), flush=True)
    Fva = fuse(Sva, best_w)

    print("\n[3/4] 逐类阈值坐标搜索（在最佳权重上）", flush=True)
    grid = np.arange(0.10, 1.00, 0.05)
    thr = d_thr.copy(); cur = evaluate(Fva, Pva, Gva, thr, nC)["Mean IoU"]
    print("  起始 %.4f (thr=%s)" % (cur, np.round(thr, 2)), flush=True)
    for it in range(4):
        improved = False
        for c in range(nC):
            best_v, best_t = cur, thr[c]
            for v in grid:
                t = thr.copy(); t[c] = v
                s = evaluate(Fva, Pva, Gva, t, nC)["Mean IoU"]
                if s > best_v:
                    best_v, best_t = s, v
            if best_t != thr[c]:
                thr[c] = best_t; cur = best_v; improved = True
                print("  pass%d cls%d -> %.2f  val %.4f" % (it, c, best_t, cur), flush=True)
        if not improved:
            break
    print("  -> 最佳阈值 %s  val mIoU %.4f" % (np.round(thr, 2), cur), flush=True)

    print("\n[4/4] 应用到 test", flush=True)
    import gc
    del Sva, Pva, Gva, Fva
    gc.collect()
    St, Pt, Gt = collect(args.weights, args.network, args.testroot, nC, args.img_size, args.num_workers)
    Ft = fuse(St, best_w)
    s_def = evaluate(Ft, Pt, Gt, d_thr, nC)
    s_tun = evaluate(Ft, Pt, Gt, thr, nC)
    print("  test 官方超参 : mIoU %.4f  mDice %.4f" % (s_def["Mean IoU"], s_def["Mean Dice"]))
    print("  test 调优超参 : mIoU %.4f  mDice %.4f  clsIoU %s" % (
        s_tun["Mean IoU"], s_tun["Mean Dice"],
        "/".join("%.3f" % s_tun["Class IoU"][c] for c in range(nC))))
    print("\n参考：ResNet38 基线 test 0.7033；论文 0.7182；SOTA(C2RM-Seg) 0.7217")


if __name__ == "__main__":
    main()
