# -*- coding: utf-8 -*-
"""论文定性对比图（参考多方法网格风格）：
6 行 = LUAD×3 + BCSS×3；6 列 = Image | GT | Single | Score avg | wVOTE | Ours。
每行统一 ROI 白框；左侧数据集标签；底部双图例。
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "research", "figs")
CACHE = "/tmp/ens"

CMAP = np.array([
    [0.89, 0.10, 0.11],   # 0 红
    [0.30, 0.69, 0.29],   # 1 绿
    [0.22, 0.49, 0.72],   # 2 蓝
    [1.00, 0.80, 0.00],   # 3 黄
    [0.96, 0.96, 0.96],   # 4 忽略（浅灰）
])
C_EDGE = "#CBD5E1"

BCSS_TAGS = ["k15_s42", "k15_s11", "k15_s21", "k15_s33", "k15_s44", "k21_s42",
             "k7_s42", "aug_s42", "bl_s42", "ch15_s42", "gsr_s42", "pv1_448",
             "pv1_e100", "pv1_ft4", "pv1_ft4lr", "pv1_s11", "pv1_s2024",
             "pv1_seed42", "pv2", "pv2s11", "pv2s21", "pv2s33", "pv2s44"]
BCSS_THR = np.asarray([0.8, 0.9, 0.8, 0.6], np.float32)
LUAD_THR = np.full(4, 0.2, np.float32)

BCSS_IDX = [1460, 1702, 628]
LUAD_IDX = [209, 31, 87]


def walk_imgs(root):
    """与 Stage1_InferDataset 一致：os.walk 的文件顺序（非字母序）。"""
    out = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".png") or f.endswith(".jpg"):
                out.append(os.path.join(r, f))
    return out


def colorize(lab):
    lab = np.clip(lab.astype(int), 0, 4)
    return CMAP[lab]


def colorize_pred(lab, gt):
    """预测面板：忽略区与 GT 一致显示为白色（仅显示用，不参与计分）。"""
    img = colorize(lab)
    img[np.asarray(gt) == 4] = CMAP[4]
    return img


def best_roi(gt, single, same, ours, size=56, stride=8):
    """选取"基线错、本方法对"最集中的窗口，用于 ROI 框。"""
    g = np.asarray(gt).astype(np.uint8)
    valid = g != 4
    good = ((single != g) | (same != g)) & (ours == g) & valid
    H, W = g.shape
    ii = np.zeros((H + 1, W + 1), np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(good.astype(np.float64), 0), 1)
    best = (-1, W // 2 - size // 2, H // 2 - size // 2)
    for y in range(0, H - size + 1, stride):
        for x in range(0, W - size + 1, stride):
            s = ii[y + size, x + size] - ii[y, x + size] - ii[y + size, x] + ii[y, x]
            if s > best[0]:
                best = (s, x, y)
    return best[1], best[2]


def gate(p, thr):
    lab = (p > thr).astype(np.float32)
    if lab.sum() == 0:
        lab[int(np.argmax(p))] = 1.0
    return lab


def member_pred(ds, tag, i, thr):
    A = np.asarray(np.load(f"{CACHE}/{ds}_{tag}_test_fused.npy", mmap_mode="r")[i],
                   dtype=np.float32)
    p = np.load(f"{CACHE}/{ds}_{tag}_test_probs.npy")[i]
    return (A * gate(p, thr)[:, None, None]).argmax(0).astype(np.uint8)


def vote_one(ds, tags, i, thr, weighted=False):
    votes = np.zeros((4, 224, 224), np.float32)
    for t in tags:
        pred = member_pred(ds, t, i, thr)
        if weighted:
            p = np.load(f"{CACHE}/{ds}_{t}_test_probs.npy")[i]
            w = p[pred]                      # 该成员在预测类上的置信度
        else:
            w = 1.0
        for k in range(4):
            votes[k] += w * (pred == k)
    return votes.argmax(0).astype(np.uint8)


def main():
    z_b = np.load(os.path.join(OUT, ".maps_bcss_test.npz"), allow_pickle=True)
    z_l = np.load(os.path.join(OUT, ".maps_luad_test.npz"), allow_pickle=True)
    g_b = np.load(f"{CACHE}/bcss_k15_s42_test_gts.npy", allow_pickle=True)
    g_l = np.load(f"{CACHE}/luad_k15_s11_test_gts.npy", allow_pickle=True)
    p_b = walk_imgs(os.path.join(ROOT, "datasets/BCSS-WSSS/test/img"))
    p_l = walk_imgs(os.path.join(ROOT, "datasets/LUAD-HistoSeg/test/img"))

    BCSS_R11 = ["aug_s42", "bl_s42", "ch15_s42", "gsr_s42", "k15_s11", "k15_s21",
                "k15_s33", "k15_s42", "k15_s44", "k21_s42", "k7_s42"]
    LUAD_G4 = ["bl_s11", "ch15_s11", "gsr_s11", "k7_s11"]

    rows = []
    for i in LUAD_IDX:
        rows.append(("LUAD", i, p_l[i], g_l[i], z_l["single"][i],
                     vote_one("luad", LUAD_G4, i, LUAD_THR),
                     z_l["Sv"][i]))
    for i in BCSS_IDX:
        rows.append(("BCSS", i, p_b[i], g_b[i],
                     member_pred("bcss", "k21_s42", i, BCSS_THR),
                     vote_one("bcss", BCSS_R11, i, BCSS_THR),
                     z_b["Sv"][i]))

    titles = ["Image", "Ground truth", "Single model",
              "Same-architecture\nensemble", "Cross-architecture\n+ voting (Ours)"]
    nrow, ncol = 6, 5
    ROI_SZ = 58

    fig = plt.figure(figsize=(12.4, 11.6), dpi=300)
    gs = fig.add_gridspec(7, ncol, height_ratios=[1, 1, 1, 0.22, 1, 1, 1],
                          left=0.070, right=0.850, top=0.955, bottom=0.045,
                          wspace=0.03, hspace=0.035)
    axes = np.empty((nrow, ncol), dtype=object)
    for r in range(3):
        for c in range(ncol):
            axes[r, c] = fig.add_subplot(gs[r, c])
    for r in range(3):
        for c in range(ncol):
            axes[3 + r, c] = fig.add_subplot(gs[4 + r, c])

    for r, (ds, i, img_path, gt, single, samearch, vote) in enumerate(rows):
        raw = np.asarray(Image.open(img_path).convert("RGB").resize((224, 224), Image.BILINEAR))
        panels = [raw, colorize(gt), colorize_pred(single, gt),
                  colorize_pred(samearch, gt), colorize_pred(vote, gt)]
        roi_xy = best_roi(gt, single, samearch, vote, size=ROI_SZ)
        for c, im in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(C_EDGE); sp.set_linewidth(0.8)
            if r == 0:
                ax.set_title(titles[c], fontsize=11, pad=8, color="#0F172A")
            rx, ry = roi_xy
            ax.add_patch(Rectangle((rx, ry), ROI_SZ, ROI_SZ,
                                   fill=False, edgecolor="white", lw=1.3))

    # 左侧数据集标签（按实际子图位置居中）
    def ycenter(r0, r1):
        p0 = axes[r0, 0].get_position(); p1 = axes[r1, 0].get_position()
        return (p0.y0 + p1.y1) / 2
    fig.text(0.040, ycenter(0, 2), "LUAD", rotation=90, va="center", ha="center",
             fontsize=13, weight="bold")
    fig.text(0.040, ycenter(3, 5), "BCSS", rotation=90, va="center", ha="center",
             fontsize=13, weight="bold")

    # 双行图例（左下对齐）
    def legend_items(names):
        return [Line2D([0], [0], marker="s", ls="", color=CMAP[k], markersize=10,
                       label=n) for k, n in enumerate(names)]
    ign = Line2D([0], [0], marker="s", ls="", markerfacecolor=CMAP[4],
                 markeredgecolor="#94A3B8", markersize=10, label="Ignore")
    luc = legend_items(["TE", "TAS", "LYM", "NEC"]) + [ign]
    bcc = legend_items(["TUM", "STR", "LYM", "NEC"]) + [ign]

    fig.legend(handles=luc, loc="center left", ncol=1, fontsize=9, frameon=False,
               bbox_to_anchor=(0.862, ycenter(0, 2)), title="LUAD", title_fontsize=10,
               handletextpad=0.5, labelspacing=0.9)
    fig.legend(handles=bcc, loc="center left", ncol=1, fontsize=9, frameon=False,
               bbox_to_anchor=(0.862, ycenter(3, 5)), title="BCSS", title_fontsize=10,
               handletextpad=0.5, labelspacing=0.9)

    out = os.path.join(OUT, "fig3_qualitative_v8.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
