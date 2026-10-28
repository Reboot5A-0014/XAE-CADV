"""生成论文所需可视化图（输出到 research/figs/）。

图 1 方法架构图（示意）
图 2 决策层共识示意（BCSS test 真实样本：某像素上成员投票 vs 打分平均）
图 3 分割结果对比（BCSS test：原图 / GT / 打分平均(模块一) / 决策共识(模块二)）
图 4 成员池族比例—增益曲线（BCSS，由表 6 数据）
图 5 逐类 IoU 柱状图（BCSS，由表 2 数据）

用法：python research/make_figs.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from scipy.cluster.hierarchy import linkage, leaves_list

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.cacg import discover, load_probs, labels_from  # noqa: E402

OUT = "research/figs"
os.makedirs(OUT, exist_ok=True)
N_CLASS = 4
THR = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
CMAP = np.array([
    [0.89, 0.10, 0.11],   # 0 TUM 红
    [0.30, 0.69, 0.29],   # 1 STR 绿
    [0.22, 0.49, 0.72],   # 2 LYM 蓝
    [1.00, 0.80, 0.00],   # 3 NEC 黄
    [1.00, 1.00, 1.00],   # 4 忽略/背景 白
])
CLS_NAME = ["TUM", "STR", "LYM", "NEC"]


def colorize(label):
    lab = np.clip(label.astype(int), 0, 4)
    img = CMAP[lab]
    # 忽略区域用浅灰
    img[lab == 4] = np.array([0.93, 0.93, 0.93])
    return img


def load_raw(name):
    p = os.path.join("datasets/BCSS-WSSS/test/img", name + ".png")
    im = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(im) / 255.0


# ------------------------------------------------------------------ 图 4 / 图 5（纯数据）
def fig4():
    fam = [2, 6, 12]                                   # 族 F 成员数（对应 13/17/23 成员池）
    mem = [13, 17, 23]
    vote = [-0.32, -0.24, +0.38]
    fbv = [+0.69, +0.21, +0.32]
    x = np.arange(len(mem))
    plt.figure(figsize=(5, 3.4), dpi=200)
    plt.axhline(0, color="gray", lw=0.8, ls="--")
    plt.plot(x, vote, "-o", label="Decision voting (VOTE)", color="#d62728")
    plt.plot(x, fbv, "-s", label="Family-balanced (FBV)", color="#1f77b4")
    for i, (a, b) in enumerate(zip(vote, fbv)):
        plt.annotate(f"{a:+.2f}", (i, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#d62728")
        plt.annotate(f"{b:+.2f}", (i, b), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#1f77b4")
    plt.xticks(x, [f"{m}\n(F:{f})" for m, f in zip(mem, fam)])
    plt.xlabel("Pool size (#members / #Family-F)")
    plt.ylabel("test mIoU gain (vs score averaging)")
    plt.title("Effect of family ratio on aggregation (BCSS)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_family_ratio_gain.png")
    plt.close()


def fig5():
    cls_m = [82.1, 75.3, 60.1, 70.4]
    cls_v = [82.4, 75.6, 60.6, 70.9]
    x = np.arange(4)
    w = 0.36
    plt.figure(figsize=(5, 3.4), dpi=200)
    b1 = plt.bar(x - w / 2, cls_m, w, label="Score averaging (baseline)", color="#9ecae1")
    b2 = plt.bar(x + w / 2, cls_v, w, label="Decision voting (CADV)", color="#d62728")
    for xi, (a, b) in enumerate(zip(cls_m, cls_v)):
        plt.annotate(f"{a:.1f}", (xi - w / 2, a), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        plt.annotate(f"{b:.1f}", (xi + w / 2, b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color="#d62728")
        plt.annotate(f"+{b-a:.1f}", (xi, max(a, b) + 3.5), ha="center", fontsize=8, color="#444444")
    plt.xticks(x, CLS_NAME)
    plt.ylim(50, 90)
    plt.ylabel("IoU (%)")
    plt.title("Per-class IoU (BCSS test)")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_per_class_iou.png")
    plt.close()


# ------------------------------------------------------------------ 图 1（示意）
def fig1():
    fig, ax = plt.subplots(figsize=(12.5, 6), dpi=200)
    ax.axis("off")
    ax.set_xlim(0, 13.4); ax.set_ylim(0, 6.6)
    FR, FF, MC, MD, IN, OL = "#dce9f7", "#e6f4e6", "#fff3d6", "#f7d6d6", "#f0f0f0", "#eaeaea"

    def box(x, y, w, h, text, fc, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc=fc, ec="#444444", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, color="#555555"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.4))

    ax.text(4.2, 6.3, "Cross-architecture member pool  (trained/adapted with image-level labels only)",
            ha="center", fontsize=10, weight="bold", color="#222222")
    box(0.3, 2.9, 1.7, 0.9, "Input image\n224 x 224", IN, fs=9)
    box(2.6, 4.4, 3.4, 1.5, "Family R: ResNet38 CNN backbones\n(multi-seed / HFRM variants,\nend-to-end training)", FR)
    box(2.6, 1.4, 3.4, 1.5, "Family F: pathology foundation models\nPhikon v1 / v2\n(frozen backbone + adapters)", FF)
    box(6.7, 4.4, 2.4, 1.5, "Per-member CAM\nper-class min-max\n+ 0.6/0.2/0.2 fusion", MC, fs=8)
    box(6.7, 1.4, 2.4, 1.5, "Image-level gating\ng_k = 1[p_k > tau_k]", MC, fs=8.5)
    box(9.9, 4.4, 3.1, 1.5, "Score averaging (baseline)\nA_bar, p_bar -> argmax", MD, fs=8.5)
    box(9.9, 1.3, 3.1, 1.5, "CADV (ours)\ndecision-level voting\nvote_k = sum_m w_m 1[S_m = k]\n-> argmax", MD, fs=8.5)
    box(9.9, 2.95, 3.1, 0.8, "Pixel-level label S_hat", OL, fs=9)

    arrow(2.0, 3.35, 2.6, 5.0)         # input -> Family R
    arrow(2.0, 3.35, 2.6, 2.0)         # input -> Family F
    arrow(6.0, 5.1, 6.7, 5.1)          # R -> CAM
    arrow(6.0, 2.0, 6.7, 2.0)          # F -> gating
    arrow(9.1, 5.1, 9.9, 5.1)          # CAM -> Module1
    arrow(9.1, 2.0, 9.9, 2.0)          # gating -> Module2
    arrow(11.45, 4.4, 11.45, 3.75)     # Module1 -> output
    arrow(11.45, 2.8, 11.45, 2.95)     # Module2 -> output
    ax.text(5.3, 5.78, "multi-scale CAM", fontsize=8, color="#777777")
    ax.text(5.3, 0.72, "image-level presence", fontsize=8, color="#777777")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_architecture.png", bbox_inches="tight")
    plt.close()

# ------------------------------------------------------------------ 图 2 / 图 3（真实数据）
def build_maps(cache="research/figs/.maps_bcss_test.npz"):
    """在 BCSS test 上计算 打分平均预测 与 决策共识预测，并缓存。"""
    if os.path.exists(cache):
        z = np.load(cache)
        return z["Sm"], z["Sv"]
    mt = discover("/tmp/ens", "bcss", "test")
    P, gts = load_probs([p for _, _, p in mt])
    M, N, C = P.shape
    A0 = np.load(mt[0][1], mmap_mode="r")
    H, W = A0.shape[2], A0.shape[3]
    mean_acc = np.zeros((N, C, H, W), dtype=np.float32)
    votes = np.zeros((N, C, H, W), dtype=np.float32)
    block = 64
    for m, (t, f, _) in enumerate(mt):
        lab = labels_from(P[m], THR)
        A = np.load(f, mmap_mode="r")
        for st in range(0, N, block):
            en = min(st + block, N)
            a = np.asarray(A[st:en], dtype=np.float32)
            pred = (a * lab[st:en, :, None, None]).argmax(axis=1)          # [b,H,W]
            mean_acc[st:en] += a
            flat = votes[st:en].reshape(en - st, C, H * W)
            idx = pred.reshape(en - st, 1, H * W)
            cur = np.take_along_axis(flat, idx, axis=1)
            np.put_along_axis(flat, idx, cur + 1.0, axis=1)
        print(f"  [{m+1}/{M}] {t}", flush=True)
    mean_acc /= M
    Sm = (mean_acc * labels_from(P.mean(0), THR)[:, :, None, None]).argmax(1).astype(np.uint8)
    Sv = votes.argmax(1).astype(np.uint8)
    np.savez_compressed(cache, Sm=Sm, Sv=Sv)
    return Sm, Sv


def pick_cases(Sm, Sv, gts, k=3, min_acc=0.80):
    """选出"整体预测较准（vote 准确率>=min_acc）且投票修复像素最多"的样本。"""
    cand = []
    for i, g in enumerate(gts):
        g = g.astype(np.uint8)
        m = g != 4
        if m.sum() == 0:
            continue
        av = (Sv[i][m] == g[m]).mean()
        if av < min_acc:
            continue
        fix = int(((Sm[i] != g) & (Sv[i] == g) & m).sum())
        cand.append((fix, av, i))
    cand.sort(reverse=True)
    return [i for _, _, i in cand[:k]]


def fig3(Sm, Sv, gts):
    idxs = pick_cases(Sm, Sv, gts, 3)
    from tool.GenDataset import Stage1_InferDataset
    ds = Stage1_InferDataset(data_path="datasets/BCSS-WSSS/test/img/", img_size=224)
    fig, axes = plt.subplots(3, 4, figsize=(9.2, 7.0), dpi=200)
    titles = ["Image", "GT", "Score averaging (baseline)", "Decision voting (CADV)"]
    for r, i in enumerate(idxs):
        name = ds[i][0]
        raw = load_raw(name)
        g = gts[i].astype(np.uint8)
        axes[r, 0].imshow(raw); axes[r, 0].set_ylabel(f"# {i}", fontsize=8)
        axes[r, 1].imshow(colorize(g))
        axes[r, 2].imshow(colorize(Sm[i]))
        axes[r, 3].imshow(colorize(Sv[i]))
        # 标出"投票修复、平均出错"的像素
        fix = (Sm[i] != g) & (Sv[i] == g) & (g != 4)
        axes[r, 3].imshow(np.where(fix, 1.0, np.nan), cmap="spring", vmin=0, vmax=1, alpha=0.75)
        for c in range(4):
            axes[0, c].set_title(titles[c], fontsize=9)
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    # 图例
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=CMAP[c], label=CLS_NAME[c], markersize=8) for c in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(f"{OUT}/fig3_qualitative.png")
    plt.close()


def fig2(Sm, Sv, gts):
    """在一个"平均出错、投票正确"的样本上，展示该像素的成员投票细节。"""
    idxs = pick_cases(Sm, Sv, gts, 1)
    i = idxs[0]
    g = gts[i].astype(np.uint8)
    fix = (Sm[i] != g) & (Sv[i] == g) & (g != 4)
    ys, xs = np.where(fix)
    y, x = int(np.median(ys)), int(np.median(xs))
    mt = discover("/tmp/ens", "bcss", "test")
    P, _ = load_probs([p for _, _, p in mt])
    thr = THR
    member_cls, member_w = [], []
    for m, (t, f, _) in enumerate(mt):
        lab = labels_from(P[m, i][None], thr)[0]
        A = np.asarray(np.load(f, mmap_mode="r")[i], dtype=np.float32)
        pred = (A * lab[:, None, None]).argmax(0)
        member_cls.append(int(pred[y, x]))
        member_w.append(float(P[m, i, pred[y, x]]))
    member_cls = np.array(member_cls)
    counts = np.array([(member_cls == c).sum() for c in range(4)])
    A_mean = np.zeros((4, 224, 224), np.float32)
    for (t, f, _) in mt:
        A_mean += np.asarray(np.load(f, mmap_mode="r")[i], dtype=np.float32)
    A_mean /= len(mt)
    mu = np.mean([np.load(mt[k][2] + "_probs.npy")[i] for k in range(len(mt))], axis=0)
    mean_pred = int(np.argmax(A_mean * labels_from(mu[None], thr)[0][:, None, None], axis=0)[y, x])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=200)
    from tool.GenDataset import Stage1_InferDataset
    ds = Stage1_InferDataset(data_path="datasets/BCSS-WSSS/test/img/", img_size=224)
    raw = load_raw(ds[i][0])
    axes[0].imshow(raw)
    axes[0].add_patch(plt.Rectangle((x - 1, y - 1), 2, 2, color="#00ff00", lw=1))
    axes[0].add_patch(plt.Rectangle((x - 20, y - 20), 40, 40, color="#00ff00", fill=False, lw=1.2))
    axes[0].set_title(f"Sample #{i} (green: disputed pixel)", fontsize=9)

    colors = [CMAP[c] for c in member_cls]
    axes[1].bar(np.arange(len(member_cls)), np.ones(len(member_cls)), color=colors, width=0.9)
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(np.arange(len(member_cls)))
    axes[1].set_xticklabels([f"{k+1}" for k in range(len(member_cls))], fontsize=6)
    axes[1].set_yticks([])
    axes[1].set_xlabel("member id")
    axes[1].set_title(f"Per-member decision at this pixel\navg -> {CLS_NAME[mean_pred]} (wrong), vote -> {CLS_NAME[int(counts.argmax())]} (GT={CLS_NAME[int(g[y,x])]})", fontsize=8)

    axes[2].bar(np.arange(4), counts, color=[CMAP[c] for c in range(4)])
    for c in range(4):
        axes[2].text(c, counts[c] + 0.2, str(int(counts[c])), ha="center", fontsize=9)
    axes[2].set_xticks(np.arange(4)); axes[2].set_xticklabels(CLS_NAME)
    axes[2].set_ylabel("Votes")
    axes[2].set_title("Decision consensus: vote counts", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig2_decision_consensus.png")
    plt.close()


# ------------------------------------------------------------------ 图 6 / 图 7（新角度）
def fig6():
    """跨成员像素级决策一致性（BCSS test 前 300 张）：族内 vs 跨族——支撑"跨架构误差去相关"。"""
    mt = discover("/tmp/ens", "bcss", "test")
    P, gts = load_probs([p_ for _, _, p_ in mt])
    N = 300
    H = W = 224
    M = len(mt)
    S = np.stack([labels_from(P[m, :N], THR) for m in range(M)])     # [M,N,C]  float
    preds = np.empty((M, N, H, W), dtype=np.uint8)
    block = 64
    for m, (t, f, _) in enumerate(mt):
        A = np.load(f, mmap_mode="r")
        lab = S[m]
        for st in range(0, N, block):
            en = min(st + block, N)
            preds[m, st:en] = (np.asarray(A[st:en], dtype=np.float32) * lab[st:en, :, None, None]).argmax(1)
        print(f"  [agree {m+1}/{M}] {t}", flush=True)
    tags = [t for t, _, _ in mt]
    idx = [i for i, t in enumerate(tags) if not t.startswith("pv")] + [i for i, t in enumerate(tags) if t.startswith("pv")]
    nR = len([t for t in tags if not t.startswith("pv")])
    mask = np.stack([gts[i] != 4 for i in range(N)])                      # 仅前景像素
    agree = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            same = (preds[idx[i]] == preds[idx[j]]) & mask
            agree[i, j] = same.sum() / mask.sum()
    def _reorder(sub):
        if len(sub) < 3:
            return sub
        D = 1 - agree[np.ix_(sub, sub)]
        Z = linkage(D[np.triu_indices(len(sub), 1)], method="average")
        return [sub[j] for j in leaves_list(Z)]
    disp = _reorder(list(range(nR))) + _reorder(list(range(nR, M)))     # 族内按相关性聚类重排
    Ag = agree[np.ix_(disp, disp)]
    intraR, cross, intraF = Ag[:nR, :nR].mean(), Ag[:nR, nR:].mean(), Ag[nR:, nR:].mean()
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=200)
    im = ax.imshow(Ag, cmap="viridis", vmin=0, vmax=1)
    ax.axhline(nR - 0.5, color="white", lw=2); ax.axvline(nR - 0.5, color="white", lw=2)
    ax.set_xticks(range(M)); ax.set_yticks(range(M))
    ax.set_xticklabels([tags[i] for i in disp], rotation=90, fontsize=6)
    ax.set_yticklabels([tags[i] for i in disp], fontsize=6)
    ax.text((nR - 1) / 2, (nR - 1) / 2, f"{intraR:.2f}", ha="center", va="center", color="white", fontsize=12, weight="bold")
    ax.text((nR + M - 1) / 2, (nR - 1) / 2, f"{cross:.2f}", ha="center", va="center", color="white", fontsize=12, weight="bold")
    ax.text((nR + M - 1) / 2, (nR + M - 1) / 2, f"{intraF:.2f}", ha="center", va="center", color="white", fontsize=12, weight="bold")
    ax.set_xlabel("member"); ax.set_ylabel("member")
    fig.colorbar(im, ax=ax, label="per-pixel decision agreement")
    ax.set_title(f"Per-pixel decision agreement on foreground (BCSS test, {N} images)\n"
                 f"intra-R {intraR:.2f}  |  cross-family {cross:.2f}  |  intra-F {intraF:.2f}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig6_member_agreement.png")
    plt.close()
    print(f"  agreement(pixel): intraR={intraR:.3f} cross={cross:.3f} intraF={intraF:.3f}")


def fig7():
    """累积消融并列柱状图：单成员 -> 同架构池+VOTE -> 跨架构池+VOTE（BCSS / LUAD）。"""
    stages = ["Single\nmember", "Same-arch. pool\n+ VOTE", "Cross-arch. pool\n+ VOTE"]
    bcss = [70.62, 70.76, 72.37]
    luad = [77.42, 77.90, 78.57]
    x = np.arange(3); w = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    ax.bar(x - w / 2, bcss, w, label="BCSS-WSSS", color="#4a7dc9")
    ax.bar(x + w / 2, luad, w, label="LUAD-HistoSeg", color="#e8a33d")
    deltas_b = ["", "+0.14", "+1.61"]; deltas_l = ["", "+0.48", "+0.67"]
    for xi in range(3):
        ax.text(xi - w / 2, bcss[xi] + 0.5, f"{bcss[xi]:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, luad[xi] + 0.5, f"{luad[xi]:.2f}", ha="center", fontsize=8, color="#b97f1d")
        if xi > 0:
            ax.annotate(f"{deltas_b[xi]}", (xi - w / 2, bcss[xi] + 4.5), ha="center", fontsize=8, color="#4a7dc9")
            ax.annotate(f"{deltas_l[xi]}", (xi + w / 2, luad[xi] + 4.5), ha="center", fontsize=8, color="#b97f1d")
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("test mIoU (%)")
    ax.set_ylim(66, 82)
    ax.set_title("Cumulative ablation: both modules improve both benchmarks")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig7_cumulative_ablation.png")
    plt.close()


if __name__ == "__main__":
    print("fig4 ..."); fig4()
    print("fig5 ..."); fig5()
    print("fig1 ..."); fig1()
    mt = discover("/tmp/ens", "bcss", "test")
    _, gts = load_probs([mt[0][2]])
    print("build maps ..."); Sm, Sv = build_maps()
    print("fig3 ..."); fig3(Sm, Sv, gts)
    print("fig2 ..."); fig2(Sm, Sv, gts)
    print("fig6 ..."); fig6()
    print("fig7 ..."); fig7()
    print("fig8 ..."); fig8()
    print("done ->", OUT)


def _delta_of(dataset, famgroups):
    """val 判据 Δ = ρ_TP − ρ_FP（族级多数票保留率）。famgroups: [[R 成员], [F 成员], ...]。"""
    dd = {x[0]: x for x in discover("/tmp/ens", dataset, "val")}
    tags = [t for grp in famgroups for t in grp]
    mm = [dd[t] for t in tags]
    P, g = load_probs([p_ for _, _, p_ in mm])
    thr = np.full(4, 0.2, np.float32) if dataset != "bcss" else THR
    gp = np.stack([[(g_ == c).any() for c in range(4)] for g_ in g]).astype(np.float32)
    pred = (P.mean(0) > thr).astype(np.float32)
    fp = (gp == 0) & (pred == 1)
    tp = (gp == 1) & (pred == 1)
    pfams = [P[[tags.index(t) for t in grp]].mean(0) for grp in famgroups]
    nf = len(pfams)
    maj = (np.sum([(pf > thr).astype(int) for pf in pfams], 0) > nf / 2).astype(np.float32)
    rTP = float(maj[tp].mean()) if tp.any() else 0.0
    rFP = float(maj[fp].mean()) if fp.any() else 0.0
    return rTP - rFP


def fig8():
    """双面板：左=每池 打分平均 vs 决策共识 配对图；右=跨架构可分性(Δ)×族均衡 → 投票增益。"""
    pools = [
        ("BCSS R+2FM",      "bcss", 71.34, 71.02, 11, 2),
        ("BCSS R+6FM",      "bcss", 71.82, 71.58, 11, 6),
        ("BCSS 23 (R+F)",   "bcss", 71.99, 72.37, 11, 12),
        ("LUAD R+2FM",      "luad", 78.47, 77.89, 11, 2),
        ("LUAD 4R+2FM",     "luad", 79.04, 78.57, 4, 2),
        ("LUAD R+2FM+3T",   "luad", 78.38, 78.12, 11, 5),
        ("LUAD 4R+2FM+3T",  "luad", 78.10, 77.99, 4, 5),
    ]
    famgroups = {
        "BCSS R+2FM": ([["aug_s42","bl_s42","ch15_s42","gsr_s42","k15_s11","k15_s21","k15_s33","k15_s42","k15_s44","k21_s42","k7_s42"], ["pv1_s11","pv2s11"]]),
        "BCSS R+6FM": ([["aug_s42","bl_s42","ch15_s42","gsr_s42","k15_s11","k15_s21","k15_s33","k15_s42","k15_s44","k21_s42","k7_s42"], ["pv1_448","pv1_e100","pv1_ft4","pv1_ft4lr","pv1_s11","pv1_s2024"]]),
        "BCSS 23 (R+F)": None,
        "LUAD R+2FM": ([["k15_s11","k15_s42","k15_s21","k15_s33","k15_s44","bl_s11","gsr_s11","ch15_s11","k7_s11","k21_s11","k15_s45"], ["pv1_seed11","pv2s11"]]),
        "LUAD 4R+2FM": ([["bl_s11","ch15_s11","gsr_s11","k7_s11"], ["pv1_seed11","pv2s11"]]),
        "LUAD R+2FM+3T": ([["k15_s11","k15_s42","k15_s21","k15_s33","k15_s44","bl_s11","gsr_s11","ch15_s11","k7_s11","k21_s11","k15_s45"], ["pv1_seed11","pv2s11","t_cnx","t_swin","t_r50"]]),
        "LUAD 4R+2FM+3T": ([["bl_s11","ch15_s11","gsr_s11","k7_s11"], ["pv1_seed11","pv2s11","t_cnx","t_swin","t_r50"]]),
    }
    R11B = ["aug_s42","bl_s42","ch15_s42","gsr_s42","k15_s11","k15_s21","k15_s33","k15_s42","k15_s44","k21_s42","k7_s42"]
    fg = [R11B, [t for t,_,_ in discover("/tmp/ens","bcss","val") if t.startswith("pv")]]
    for name, ds, me, vo, nr, nf in pools:
        gf = famgroups[name] or fg
        delta = _delta_of(ds, gf)
        bal = min(nr, nf) / max(nr, nf)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    # --- 左：配对图 ---
    rows = []
    for name, ds, me, vo, nr, nf in pools:
        gf = famgroups[name] or fg
        delta = _delta_of(ds, gf)
        rows.append(dict(name=name, ds=ds, me=me, vo=vo, delta=delta, bal=min(nr,nf)/max(nr,nf)))
    order = sorted(rows, key=lambda r: r["vo"] - r["me"])
    for j, r in enumerate(order):
        c = "#4a7dc9" if r["ds"] == "bcss" else "#e8a33d"
        ax1.plot([0, 1], [r["me"], r["vo"]], color=c, alpha=0.85, lw=1.4, zorder=2)
        ax1.scatter([0], [r["me"]], color=c, s=45, edgecolor="k", linewidth=0.5, zorder=3)
        ax1.scatter([1], [r["vo"]], color=c, s=45, edgecolor="k", linewidth=0.5, zorder=3)
        gain = r["vo"] - r["me"]
        ax1.annotate(f"{gain:+.2f}", (1, r["vo"]), textcoords="offset points", xytext=(7, 0),
                     fontsize=7.5, color="#c0392b" if gain < 0 else "#1e8449")
        ax1.text(1.16, (r["me"] + r["vo"]) / 2, r["name"], fontsize=7, rotation=0, va="center")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["score\navg", "decision\nvote"], fontsize=8)
    ax1.set_ylabel("test mIoU (%)")
    ax1.set_ylim(69, 80.5)
    ax1.set_title("(a) Score averaging vs decision voting (per pool)")
    ax1.axvline(0.5, color="gray", ls=":", lw=0.8)
    # --- 右：Δ × 平衡 → 增益散点 ---
    xs = [r["delta"] for r in rows]; ys = [r["vo"] - r["me"] for r in rows]
    sizes = [60 + 260 * r["bal"] for r in rows]
    cols = ["#4a7dc9" if r["ds"] == "bcss" else "#e8a33d" for r in rows]
    ax2.scatter(xs, ys, s=sizes, c=cols, edgecolor="k", linewidth=0.6, zorder=3)
    for r in rows:
        ax2.annotate(r["name"], (r["delta"], r["vo"]-r["me"]), textcoords="offset points",
                     xytext=(7, 6), fontsize=7.5)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_xlabel("cross-arch FP separability  Δ (val)")
    ax2.set_ylabel("voting gain  VOTE − mean (test)")
    ax2.set_title("(b) Δ and family balance → voting gain\n(marker size = family balance)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig8_crossarch_vs_gain.png")
    plt.close()
    for r in rows:
        print(f"  {r['name']:18s} delta={r['delta']:+.3f} bal={r['bal']:.2f} gain={r['vo']-r['me']:+.2f}")
