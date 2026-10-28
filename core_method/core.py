# -*- coding: utf-8 -*-
"""核心方法（论文 §2 方法的完整、自包含实现）。

模块一 XAE（跨架构成员集成）＋ 模块二 CADV（跨架构决策共识）。
仅依赖 numpy。输入为各成员的"融合 CAM + 图像级概率 + GT"缓存
（由推理采集脚本得到），此处只给出方法本身的算法实现，
不绑定任何具体网络结构——这正是方法"即插即用"的体现。

对应论文公式：
  式(1) 多尺度 CAM 逐类 min-max 融合          -> fuse_cam_scales
  式(2) 图像级门控 + argmax 单成员决策        -> labels_from / member_decision
  式(3) 模块一 打分平均                        -> score_averaging
  式(4) 模块二 决策级共识投票                  -> decision_voting
  权重方案 VOTE / wVOTE / FBV                  -> decision_voting 参数
"""
import numpy as np


# --------------------------------------------------------------------------- #
# 0. 评估口径（与官方 tool/iouutils.scores 一致，仅统计前 4 个前景类）
# --------------------------------------------------------------------------- #
def hist_iu_dice(hist):
    """由混淆矩阵 hist[(n_class+1)^2] 计算 mIoU / mDice / 逐类 IoU。

    hist[g, p] = 真值为 g、预测为 p 的像素数（g,p ∈ {0..n_class}，
    n_class 表示"忽略"区域）。仅统计 4 个前景类。
    """
    hist = hist.copy()
    hist[4, 4] = 0  # 忽略区域不计分
    iu = np.diag(hist)[0:4] / (
        (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))[0:4]
    )
    mean_iu = float(np.nanmean(iu))
    dice = []
    for i in range(4):
        tp = np.diag(hist)[i]
        fp = hist[:, i].sum() - tp
        fn = hist[i, :].sum() - tp
        dice.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)
    return mean_iu, float(np.mean(dice)), iu


def _confusion(pred, gts, n_class):
    """累加混淆矩阵。pred/gts 为 [N,H,W] 的整型标签图。"""
    hist = np.zeros((n_class + 1, n_class + 1))
    for i, g in enumerate(gts):
        g = np.asarray(g).astype(np.int64)
        p = np.asarray(pred[i]).astype(np.int64).copy()
        p[g == n_class] = n_class          # 忽略区域剔除
        bb = (n_class + 1) * g.ravel() + p.ravel()
        hist += np.bincount(bb, minlength=(n_class + 1) ** 2
                            ).reshape(n_class + 1, n_class + 1)
    return hist


# --------------------------------------------------------------------------- #
# 1. 单成员：图像级门控 + 像素级决策  （论文式 2）
# --------------------------------------------------------------------------- #
def labels_from(probs, thr):
    """由图像级概率生成类别存在性掩码（硬门控）。

    probs: [N, C] 成员输出的图像级类别概率
    thr:   [C]   逐类阈值
    返回:  [N, C] ∈ {0,1}，全为 0 的样本回退到概率最大的类。
    """
    lab = (probs > thr).astype(np.float32)
    empty = lab.sum(axis=1) == 0
    if empty.any():
        lab[empty, np.argmax(probs[empty], axis=1)] = 1.0
    return lab


def member_decision(cam, lab):
    """单成员的像素级决策：门控后取 argmax。

    cam: [N, C, H, W] 融合 CAM（式 1）
    lab: [N, C] 图像级门控掩码（labels_from）
    返回: [N, H, W] 像素标签（0 为背景）
    """
    gated = cam.astype(np.float32) * lab[:, :, None, None]
    return gated.argmax(axis=1)


# --------------------------------------------------------------------------- #
# 2. 模块一 XAE：跨架构打分平均（论文式 3）
# --------------------------------------------------------------------------- #
def score_averaging(cam_paths, probs, thr, block=256):
    """跨架构打分融合（分块流式累加，避免一次性载入全部 CAM 导致 OOM）。

    cam_paths: [M] 各成员融合 CAM 的 .npy 路径（mmap 读取）
    probs:     [M, N, C] 各成员图像级概率
    thr:       [C]
    返回:      pred [N, H, W]
    对应式(3): A_bar = mean_m A_m, p_bar = mean_m p_m,
               S = argmax_k 1[p_bar_k > tau_k] * A_bar_k
    """
    A0 = np.load(cam_paths[0], mmap_mode="r")
    N, C, H, W = A0.shape
    acc = np.zeros((N, C, H, W), dtype=np.float64)   # 累加器
    for p in cam_paths:                     # 分块累加各成员 CAM
        A = np.load(p, mmap_mode="r")
        for st in range(0, N, block):
            acc[st:st + block] += np.asarray(A[st:st + block], dtype=np.float32)
    mean_cam = (acc / len(cam_paths)).astype(np.float32)
    mean_prob = probs.mean(axis=0)                          # [N,C]
    lab = labels_from(mean_prob, thr)                       # [N,C]
    pred = member_decision(mean_cam, lab)
    del mean_cam, acc
    return pred


# --------------------------------------------------------------------------- #
# 3. 模块二 CADV：跨架构决策共识投票（论文式 4）
# --------------------------------------------------------------------------- #
def decision_voting(cam_paths, probs, thr, n_class,
                    weighted=False, member_w=None, block=64):
    """决策级共识：各成员先独立决策（门控+argmax），再像素级（加权）多数投票。

    cam_paths: [M] 各成员融合 CAM 的 .npy 路径（mmap 读取）
    probs:     [M, N, C]
    thr:       [C]
    weighted:  False → 每成员 1 票（VOTE，默认）；
               True  → 按成员对自身预测类的置信度加权（wVOTE）。
    member_w:  [M] 每成员再乘一档权重，用于族平衡 FBV = 1/族大小；
               族内成员数 |F(m)| 越大权重越小，使各族总票数相等。
    返回:      pred [N, H, W]
    对应式(4): vote_k(u) = sum_m w_m * 1[S_m(u) = k],
               S(u) = argmax_k vote_k(u)
    """
    A0 = np.load(cam_paths[0], mmap_mode="r")
    N, _, H, W = A0.shape
    votes = np.zeros((N, n_class, H, W), dtype=np.float32)
    for m, p in enumerate(cam_paths):
        A = np.load(p, mmap_mode="r")
        lab = labels_from(probs[m], thr).astype(np.float32)   # 成员自身门控
        for st in range(0, N, block):
            en = min(st + block, N)
            b = en - st
            a = np.asarray(A[st:en], dtype=np.float32)
            pred = (a * lab[st:en, :, None, None]).argmax(axis=1)  # S_m [b,H,W]
            w = np.ones((b, H, W), dtype=np.float32)
            if weighted:                   # wVOTE：取成员在预测类上的概率
                w = np.take_along_axis(
                    probs[m, st:en], pred.reshape(b, -1), axis=1
                ).reshape(b, H, W)
            if member_w is not None:       # FBV：族平衡权重
                w = w * float(member_w[m])
            flat = votes[st:en].reshape(b, n_class, H * W)
            idx = pred.reshape(b, 1, H * W)
            cur = np.take_along_axis(flat, idx, axis=1)
            np.put_along_axis(flat, idx, cur + w.reshape(b, 1, H * W), axis=1)
    return votes.argmax(axis=1).astype(np.int64)      # argmax_k vote_k


# --------------------------------------------------------------------------- #
# 4. 便捷封装：对一组成员缓存求 mIoU / mDice / 逐类 IoU
# --------------------------------------------------------------------------- #
def evaluate(pred, gts, n_class):
    """pred [N,H,W] 与 GT 列表 → (mIoU, mDice, 逐类IoU)。"""
    return hist_iu_dice(_confusion(pred, gts, n_class))


def family_balanced_weights(tags):
    """由成员 tag 生成 FBV 权重：pv* 记为族 F，其余记为族 R，w = 1/族大小。"""
    fam = [t.startswith("pv") for t in tags]
    nF, nR = sum(fam), len(fam) - sum(fam)
    return [1.0 / nF if f else 1.0 / nR for f in fam]


# --------------------------------------------------------------------------- #
# 5. 最小示例（从缓存加载成员，对比 mean / VOTE / wVOTE / FBV）
# --------------------------------------------------------------------------- #
def demo(cache_dir, dataset, split, tags, thr, n_class=4):
    """从 {dataset}_{tag}_{split}_{fused,probs,gts}.npy 缓存评估四种聚合。"""
    cam_paths = [f"{cache_dir}/{dataset}_{t}_{split}_fused.npy" for t in tags]
    probs, gts = [], None
    for t in tags:
        pre = f"{cache_dir}/{dataset}_{t}_{split}"
        probs.append(np.load(pre + "_probs.npy").astype(np.float32))
        if gts is None:
            gts = list(np.load(pre + "_gts.npy", allow_pickle=True))
    probs = np.stack(probs, axis=0)                            # [M,N,C]

    out = {}
    pred = score_averaging(cam_paths, probs, thr)                             # mean
    out["mean"] = evaluate(pred, gts, n_class)
    pred = decision_voting(cam_paths, probs, thr, n_class)                    # VOTE
    out["VOTE"] = evaluate(pred, gts, n_class)
    pred = decision_voting(cam_paths, probs, thr, n_class, weighted=True)     # wVOTE
    out["wVOTE"] = evaluate(pred, gts, n_class)
    pred = decision_voting(cam_paths, probs, thr, n_class,
                           member_w=family_balanced_weights(tags))            # FBV
    out["FBV"] = evaluate(pred, gts, n_class)
    return out


if __name__ == "__main__":
    # 用法示例（需先有成员缓存；阈值与基线一致）
    BCSS_THR = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
    LUAD_THR = np.full(4, 0.2, dtype=np.float32)
    # res = demo("/tmp/ens", "bcss", "test",
    #            tags=["k15_s42", "pv2", ...], thr=BCSS_THR)
    # for k, v in res.items():
    #     print(f"{k:5s} mIoU={v[0]*100:.2f} mDice={v[1]*100:.2f}")
