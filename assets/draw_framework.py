# -*- coding: utf-8 -*-
"""Draw the XAE-CADV framework overview (for README / paper supplement).

(a) Overall inference pipeline
(b) XAE: cross-architecture member-pool construction (Eq.1)
(c) CADV: pixel-wise decision consensus (Eq.7-8; VOTE/wVOTE/FBV = Eq.9-14)

Equation numbers follow the paper:
  Eq.1 pool; Eq.2 per-class min-max; Eq.3 multi-scale fusion; Eq.4-5 image-level
  gate; Eq.6 member decision; Eq.7 vote accumulation; Eq.8 argmax;
  Eq.9-10 VOTE; Eq.11-12 wVOTE; Eq.13-14 FBV.

Usage:  python assets/draw_framework.py   ->  assets/framework.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
C_EDGE = "#334155"; C_R = "#2563EB"; C_F = "#16A34A"
C_XAE = "#7C3AED"; C_CADV = "#EA580C"; C_SUB = "#475569"


def box(ax, x, y, w, h, fc, ec, lw=1.6, r=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.15,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=1))


def txt(ax, x, y, s, fs=10, c="#0F172A", w="normal", ha="center", va="center", ls=1.35):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c, weight=w, linespacing=ls)


def arrow(ax, p1, p2, color=C_EDGE, lw=2.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18,
                 lw=lw, color=color))


fig, ax = plt.subplots(figsize=(16, 8.2), dpi=200)
ax.set_xlim(0, 168); ax.set_ylim(0, 84); ax.axis("off")
fig.patch.set_facecolor("white")

txt(ax, 84, 80.6, "XAE-CADV: Cross-Architecture Ensemble and Decision Voting",
    fs=16, w="bold")
txt(ax, 84, 76.6, "Inference-stage aggregation of existing single-stage CAM members "
    "trained with image-level labels only", fs=9.5, c=C_SUB)

# ---------------- (a) overall inference pipeline ----------------
txt(ax, 3.5, 72.0, "(a) Overall inference pipeline", fs=11, w="bold", ha="left")
yb, hb = 56, 15
box(ax, 3, yb, 18, hb, "#E2E8F0", C_EDGE)
txt(ax, 12, yb + 9.6, "Input image", fs=9.5, w="bold")
txt(ax, 12, yb + 6.2, "$x_i$", fs=12)
txt(ax, 12, yb + 2.8, "image-level label", fs=7.8, c=C_SUB)

box(ax, 27, yb, 33, hb, "#F5F3FF", C_XAE)
txt(ax, 43.5, yb + 12.4, "Module I: XAE  (Eq.1)", fs=9.8, w="bold", c=C_XAE)
txt(ax, 43.5, yb + 9.3, "$\\mathcal{M}=\\mathcal{M}_R\\cup\\mathcal{M}_F$", fs=10)
txt(ax, 43.5, yb + 6.0, "$\\mathcal{M}_R$: ResNet38 CNN members", fs=8.2)
txt(ax, 43.5, yb + 3.5, "$\\mathcal{M}_F$: Phikon v1 / Phikon-v2 PFMs", fs=8.2)
txt(ax, 43.5, yb + 1.3, "(cross-architecture pool)", fs=7.6, c=C_SUB)

box(ax, 66, yb - 2, 42, hb + 4, "#F8FAFC", C_EDGE, lw=1.4)
txt(ax, 87, yb + 13.6, "Per-member shared inference  ($m=1\\ldots M$)", fs=9.0, w="bold")
txt(ax, 87, yb + 10.4, "3$\\times$ TTA: align + inverse-flip $\\rightarrow$ average", fs=8.0)
txt(ax, 87, yb + 7.8, "Eq.2  per-class min-max normalization", fs=8.0)
txt(ax, 87, yb + 5.2, "Eq.3  multi-scale fusion 0.6/0.2/0.2", fs=8.0)
txt(ax, 87, yb + 2.6, "Eq.4-5  image-level gate $g^{(m)}$ (fallback)", fs=8.0)
txt(ax, 87, yb + 0.2, "Eq.6  member decision $\\hat{S}^{(m)}(u)$", fs=8.0)

box(ax, 114, yb - 2, 40, hb + 4, "#FFF7ED", C_CADV)
txt(ax, 134, yb + 13.6, "Module II: CADV  (Eq.7-8)", fs=9.8, w="bold", c=C_CADV)
txt(ax, 134, yb + 10.9, "pixel-wise decision voting", fs=8.4)
chip_y, ch = yb + 3.0, 6.0
box(ax, 116.5, chip_y, 11.5, ch, "#FFFFFF", C_CADV, lw=1.1)
txt(ax, 122.2, chip_y + 3.0, "VOTE\nEq.9-10", fs=7.6)
box(ax, 129.0, chip_y, 12.5, ch, "#FFFFFF", C_CADV, lw=1.1)
txt(ax, 135.2, chip_y + 3.0, "wVOTE\nEq.11-12", fs=7.6)
box(ax, 143.0, chip_y, 9.5, ch, "#FFFFFF", C_CADV, lw=1.1)
txt(ax, 147.7, chip_y + 3.0, "FBV\nEq.13-14", fs=7.6)
txt(ax, 134, yb + 0.7, "all parameter-free", fs=7.4, c=C_SUB)

box(ax, 160, yb + 2.5, 7.5, 10, "#F1F5F9", C_EDGE)
txt(ax, 163.7, yb + 7.5, "Output\n$\\hat{S}_i$", fs=9.0)

arrow(ax, (21, yb + hb / 2), (27, yb + hb / 2))
arrow(ax, (60, yb + hb / 2), (66, yb + hb / 2))
arrow(ax, (108, yb + hb / 2), (114, yb + hb / 2))
arrow(ax, (154, yb + 7.5), (160, yb + 7.5))

# ---------------- (b) XAE pool ----------------
txt(ax, 3.5, 45.5, "(b) XAE: cross-architecture member-pool construction (Eq.1)",
    fs=11, w="bold", ha="left")
box(ax, 3, 10, 78, 32, "#FBFBFD", C_EDGE, lw=1.4)
box(ax, 6, 20, 35, 18, "#EFF6FF", C_R)
txt(ax, 23.5, 35.2, "Family $\\mathcal{M}_R$: ResNet38 CNN members", fs=9.2, w="bold", c=C_R)
txt(ax, 23.5, 31.8, "ImageNet-initialized, trained end-to-end", fs=7.8)
txt(ax, 23.5, 29.2, "multi-seed / HFRM kernel / GSR-CH variants", fs=7.8)
txt(ax, 23.5, 26.6, "convolution locality: local texture cues", fs=7.8)
txt(ax, 23.5, 23.2, "BCSS: 11 members  |  LUAD: 4 members", fs=8.0, w="bold")
box(ax, 44, 20, 34, 18, "#F0FDF4", C_F)
txt(ax, 61, 35.2, "Family $\\mathcal{M}_F$: Phikon v1 / v2 PFMs", fs=9.2, w="bold", c=C_F)
txt(ax, 61, 31.8, "frozen backbone + lightweight adapters", fs=7.8)
txt(ax, 61, 29.2, "1$\\times$1 proj + 2$\\times$ transposed conv $\\rightarrow$ CAM head", fs=7.8)
txt(ax, 61, 26.6, "self-attention: global pathology priors", fs=7.8)
txt(ax, 61, 23.2, "BCSS: 12 members  |  LUAD: 2 members", fs=8.0, w="bold")
txt(ax, 42, 14.0, "Different inductive biases $\\rightarrow$ decorrelated errors; "
    "all members share one inference interface.", fs=8.2, c=C_SUB)

# ---------------- (c) CADV ----------------
txt(ax, 88.5, 45.5, "(c) CADV: pixel-wise decision consensus (Eq.7-8)",
    fs=11, w="bold", ha="left")
box(ax, 86, 10, 78, 32, "#FFFBF5", C_CADV, lw=1.4)
txt(ax, 125, 38.2, "Each member decides independently (Eq.6)", fs=8.4)
txt(ax, 125, 35.2, "$\\rightarrow$ accumulate class votes at every pixel (Eq.7) "
    "$\\rightarrow$ $\\arg\\max$ over classes (Eq.8)", fs=8.4)
box(ax, 90, 20, 22, 10, "#FFFFFF", C_CADV, lw=1.2)
txt(ax, 101, 26.8, "VOTE  (Eq.9-10)", fs=8.0, w="bold", c=C_CADV)
txt(ax, 101, 23.6, "$w_m=1$", fs=9.0)
box(ax, 114, 20, 22, 10, "#FFFFFF", C_CADV, lw=1.2)
txt(ax, 125, 26.8, "wVOTE  (Eq.11-12)", fs=8.0, w="bold", c=C_CADV)
txt(ax, 125, 23.6, "$w_m=p^{(m)}_{\\hat{S}^{(m)}(u)}$", fs=9.0)
box(ax, 138, 20, 22, 10, "#FFFFFF", C_CADV, lw=1.2)
txt(ax, 149, 26.8, "FBV  (Eq.13-14)", fs=8.0, w="bold", c=C_CADV)
txt(ax, 149, 23.6, "$w_m=1/n_{\\mathcal{F}(m)}$", fs=9.0)
txt(ax, 125, 16.0, "Parameter-free: no extra training, no forward pass beyond member predictions.",
    fs=8.2, c=C_SUB)
txt(ax, 125, 12.8, "Corrects member-specific errors when the remaining members vote correctly.",
    fs=8.2, c=C_SUB)

plt.tight_layout()
out = os.path.join(HERE, "framework.png")
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)