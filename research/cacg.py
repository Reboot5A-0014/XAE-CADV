"""CACG: Cross-Architecture Consensus Gating（跨架构一致性门控）—— 离线评估。

在"打分层平均集成"（09 文档 §3）之上新增的模块，只用单模型拿不到的**跨成员一致性信号**。
成员缓存为 .npy（由 research/cacg_collect.py 采集），可 mmap，支持多趟读取。

模块组成（可单独开关，用于消融）：
  A. 基线集成        : 等权融合 CAM + 官方硬门控
  B. PWF 概率加权融合 : Ā^pw_k = Σ_m (p_k^(m))^γ A_k^(m) / Σ_m (p_k^(m))^γ
  C. CRW 共识重加权   : Ã_k = lab_k · r_k^a · κ_k^b · Ā_k
  D. VOTE 决策层共识  : 各成员独立门控+argmax 后做（加权）多数投票

其中 μ_k = mean_m p_k^(m)，σ_k = std_m p_k^(m)，r_k = mean_m 1[p_k^(m) > τ_k]，
κ_k = mean_m cos(A_k^(m), Ā_k)（跨成员 CAM 空间一致性）。

用法：
  python research/cacg.py --dataset bcss --dir /tmp/ens
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.ensemble_ens import hist_iu_dice, labels_from  # noqa: E402


def discover(dirpath, dataset, split, exclude=()):
    fs = sorted(glob.glob(f"{dirpath}/{dataset}_*_{split}_fused.npy"))
    out = []
    for f in fs:
        t = os.path.basename(f)[len(dataset) + 1: -len(f"_{split}_fused.npy")]
        if t not in exclude:
            out.append((t, f, f[: -len("_fused.npy")]))
    return out


def load_probs(pref_list):
    ps, gts = [], None
    for pref in pref_list:
        ps.append(np.load(pref + "_probs.npy").astype(np.float32))
        if gts is None:
            gts = list(np.load(pref + "_gts.npy", allow_pickle=True))
    return np.stack(ps, axis=0), gts                    # [M,N,C]


def mean_cam(members, block=256):
    A0 = np.load(members[0][1], mmap_mode="r")
    N, shape = A0.shape[0], A0.shape[1:]
    out = np.zeros((N,) + tuple(shape), dtype=np.float32)
    for m, (t, f, _) in enumerate(members):
        A = np.load(f, mmap_mode="r")
        for st in range(0, N, block):
            en = min(st + block, N)
            out[st:en] += np.asarray(A[st:en], dtype=np.float32)
        print(f"  [mean {m+1}/{len(members)}] {t}", flush=True)
    out /= len(members)
    return out


def kappa_of(members, mc, block=256):
    """κ_k = mean_m cos(A_k^(m), Ā_k)，返回 [N,C]。"""
    N = mc.shape[0]
    kappa = np.zeros((N, mc.shape[1]), dtype=np.float64)
    nb = np.sqrt((mc.astype(np.float64) ** 2).sum(axis=(2, 3))) + 1e-8
    for m, (t, f, _) in enumerate(members):
        A = np.load(f, mmap_mode="r")
        for st in range(0, N, block):
            en = min(st + block, N)
            a = np.asarray(A[st:en], dtype=np.float32)
            num = (a * mc[st:en]).sum(axis=(2, 3)).astype(np.float64)
            na = np.sqrt((a.astype(np.float64) ** 2).sum(axis=(2, 3))) + 1e-8
            kappa[st:en] += num / (na * nb[st:en])
        print(f"  [kappa {m+1}/{len(members)}] {t}", flush=True)
    kappa /= len(members)
    return np.clip(kappa, 0.0, 1.0).astype(np.float32)


def pw_cam_of(members, P, gamma, block=256):
    A0 = np.load(members[0][1], mmap_mode="r")
    N, shape = A0.shape[0], A0.shape[1:]
    out = np.zeros((N,) + tuple(shape), dtype=np.float32)
    for m, (t, f, _) in enumerate(members):
        A = np.load(f, mmap_mode="r")
        for st in range(0, N, block):
            en = min(st + block, N)
            a = np.asarray(A[st:en], dtype=np.float32)
            q = np.power(np.clip(P[m, st:en], 1e-6, 1.0), gamma).astype(np.float32)
            out[st:en] += a * q[:, :, None, None]
    wsum = np.power(np.clip(P, 1e-6, 1.0), gamma).sum(axis=0)
    out /= wsum[:, :, None, None]
    return out


def evaluate_gated(cam, lab, gts, n_class, block=128):
    N = len(gts)
    hist = np.zeros((n_class + 1, n_class + 1))
    labf = np.asarray(lab, dtype=np.float32)
    for st in range(0, N, block):
        en = min(st + block, N)
        c = cam[st:en].astype(np.float32) * labf[st:en, :, None, None]
        pred = c.argmax(axis=1).astype(np.int64)
        g = np.stack([gts[i] for i in range(st, en)]).astype(np.int64)
        pred = pred.copy()
        pred[g == n_class] = n_class
        b = (n_class + 1) * g.ravel() + pred.ravel()
        hist += np.bincount(b, minlength=(n_class + 1) ** 2).reshape(n_class + 1, n_class + 1)
    return hist_iu_dice(hist)


def vote_ensemble(members, P, thr, gts, n_class, weighted=False, block=64, member_w=None):
    """决策层共识：各成员独立门控+argmax 得像素标签，再做（加权）多数投票。

    weighted=False → 每成员 1 票；weighted=True → 按该成员在预测类上的概率加权；
    member_w 给定 [M] 时再乘一档每成员权重（用于族间平衡，如 1/族大小）。
    """
    A0 = np.load(members[0][1], mmap_mode="r")
    N, H, W = A0.shape[0], A0.shape[2], A0.shape[3]
    votes = np.zeros((N, n_class, H, W), dtype=np.float32)
    for m, (t, f, _) in enumerate(members):
        A = np.load(f, mmap_mode="r")
        lab = labels_from(P[m], thr).astype(np.float32)
        for st in range(0, N, block):
            en = min(st + block, N)
            b = en - st
            a = np.asarray(A[st:en], dtype=np.float32)
            pred = (a * lab[st:en, :, None, None]).argmax(axis=1)          # [b,H,W]
            w = np.ones((b, H, W), dtype=np.float32)
            if weighted:
                w = np.take_along_axis(P[m, st:en], pred.reshape(b, -1), axis=1).reshape(b, H, W)
            if member_w is not None:
                w = w * float(member_w[m])
            flat = votes[st:en].reshape(b, n_class, H * W)                 # view
            idx = pred.reshape(b, 1, H * W)
            cur = np.take_along_axis(flat, idx, axis=1)
            np.put_along_axis(flat, idx, cur + w.reshape(b, 1, H * W), axis=1)
        print(f"  [vote {m+1}/{len(members)}] {t}", flush=True)
    final = votes.argmax(axis=1).astype(np.int64)
    hist = np.zeros((n_class + 1, n_class + 1))
    for i, g in enumerate(gts):
        g = g.astype(np.int64)
        p = final[i].copy()
        p[g == n_class] = n_class
        bb = (n_class + 1) * g.ravel() + p.ravel()
        hist += np.bincount(bb, minlength=(n_class + 1) ** 2).reshape(n_class + 1, n_class + 1)
    return hist_iu_dice(hist)


def gt_presence(gts, n_class):
    return np.stack([[(g == c).any() for c in range(n_class)] for g in gts]).astype(np.float32)


def apply_veto(lab0, r, kappa, rho_r, rho_k, prob):
    """共识硬否决：共识率 r 或空间一致性 κ 低于阈值的类被否决；全否决则退化为 argmax(prob)。"""
    keep = (r >= rho_r) & (kappa >= rho_k)
    lab = lab0 * keep.astype(np.float32)
    e = lab.sum(axis=1) == 0
    if e.any():
        lab[e, np.argmax(prob[e], axis=1)] = 1.0
    return lab


def _op(stack, name):
    """stack: [M,b,C,H,W]（各成员已是逐类 min-max 的融合 CAM）→ [b,C,H,W]。"""
    if name == "mean":
        return stack.mean(axis=0)
    if name == "median":
        return np.median(stack, axis=0)
    if name == "p25":
        return np.percentile(stack, 25, axis=0)
    if name == "trim1":
        return (stack.sum(axis=0) - stack.min(axis=0)) / (stack.shape[0] - 1)
    raise ValueError(name)


def consensus_ops_eval(members, lab, gts, n_class, ops, block=48):
    """逐像素跨成员共识融合：对每个像素/类取跨成员的中位数/分位数等，替代等权平均。"""
    A0 = np.load(members[0][1], mmap_mode="r")
    N, H, W = A0.shape[0], A0.shape[2], A0.shape[3]
    hists = {op: np.zeros((n_class + 1, n_class + 1)) for op in ops}
    for st in range(0, N, block):
        en = min(st + block, N)
        b = en - st
        stack = np.empty((len(members), b, n_class, H, W), dtype=np.float32)
        for m, (t, f, _) in enumerate(members):
            stack[m] = np.asarray(np.load(f, mmap_mode="r")[st:en], dtype=np.float32)
        g = np.stack([gts[i] for i in range(st, en)]).astype(np.int64)
        for op in ops:
            cam = _op(stack, op) * lab[st:en, :, None, None]
            pred = cam.argmax(axis=1).astype(np.int64)
            pred[g == n_class] = n_class
            bb = (n_class + 1) * g.ravel() + pred.ravel()
            hists[op] += np.bincount(bb, minlength=(n_class + 1) ** 2).reshape(n_class + 1, n_class + 1)
        del stack
        print(f"  [consensus] {en}/{N}", flush=True)
    return {op: hist_iu_dice(h) for op, h in hists.items()}


def auc(x, y):
    order = np.argsort(x)
    y = y[order]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bcss")
    ap.add_argument("--dir", default="/tmp/ens")
    ap.add_argument("--n_class", default=4, type=int)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--tags", default="", help="逗号分隔，仅评估列出的成员（冒烟测试用）")
    ap.add_argument("--block", default=256, type=int)
    ap.add_argument("--pw_gammas", default="1,2,4")
    ap.add_argument("--report", default="both", choices=["val", "test", "both"])
    args = ap.parse_args()
    n = args.n_class
    thr = (np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32) if args.dataset == "bcss"
           else np.full(n, 0.2, dtype=np.float32))
    excl = tuple(x for x in args.exclude.split(",") if x)

    mv = discover(args.dir, args.dataset, "val", excl)
    mt = discover(args.dir, args.dataset, "test", excl)
    if args.tags:
        want = set(x for x in args.tags.split(",") if x)
        mv = [m for m in mv if m[0] in want]
        mt = [m for m in mt if m[0] in want]
    tags = [t for t, _, _ in mv]
    print(f"成员（{len(tags)}）: {tags}\n", flush=True)
    Pv, gv = load_probs([p for _, _, p in mv])
    Pt, gt = load_probs([p for _, _, p in mt])

    def feats(P):
        return P.mean(0), P.std(0), (P > thr.reshape(1, 1, -1)).mean(0), P.min(0), P.max(0)

    best = {}

    # ---------------- VAL ----------------
    if args.report in ("val", "both"):
        print("=== VAL ===", flush=True)
        mcv = mean_cam(mv, args.block)
        muv, sdv, rv, _, _ = feats(Pv)
        kpv = kappa_of(mv, mcv, args.block)
        lab0v = labels_from(muv, thr)
        b_v = evaluate_gated(mcv, lab0v, gv, n)
        print(f"[VAL ] baseline 集成                mIoU={b_v[0]*100:6.2f}  mDice={b_v[1]*100:6.2f}", flush=True)
        gpv = gt_presence(gv, n)
        isfm = np.array([t.startswith("pv") for t in tags])
        famgapv = np.abs(Pv[isfm].mean(0) - Pv[~isfm].mean(0)) if isfm.any() and (~isfm).any() else np.zeros_like(muv)
        _o = np.sort(muv, axis=1)
        marginv = np.where(muv >= _o[:, -1:], muv - _o[:, -2:-1], muv - _o[:, -1:])
        predv = (muv > thr).astype(np.float32)
        fp = ((gpv == 0) & (predv == 1))
        tp = ((gpv == 1) & (predv == 1))
        print(f"       实例级: TP={int(tp.sum())} FN={int(((gpv==1)&(predv==0)).sum())} FP={int(fp.sum())}")
        for name, s in [("mu", muv), ("sigma", sdv), ("1-r", 1 - rv), ("1-kappa", 1 - kpv),
                        ("sigma/mu", sdv / (muv + 1e-6)), ("famgap", famgapv), ("margin", marginv)]:
            y = np.concatenate([fp.ravel().astype(int), tp.ravel().astype(int)])
            x = np.concatenate([s[fp].ravel(), s[tp].ravel()])
            print(f"       FP vs TP AUC({name:9s}) = {auc(x, y):.3f}")
        grid = [0.0, 0.5, 1.0, 2.0]
        res = []
        for a in grid:
            for b in grid:
                mm = evaluate_gated(mcv, lab0v * (rv ** a) * (kpv ** b), gv, n)
                res.append((mm[0], a, b))
        res.sort(reverse=True)
        print("       ── CRW top5(val) ──")
        for s, a, b in res[:5]:
            print(f"       a={a:<4} b={b:<4} mIoU={s*100:6.2f} ({s*100-b_v[0]*100:+.2f})")
        best["crw"] = (res[0][1], res[0][2])
        vres = []
        for qr in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
            for qk in [0.0, 0.2, 0.4, 0.6]:
                mm = evaluate_gated(mcv, apply_veto(lab0v, rv, kpv, qr, qk, muv), gv, n)
                vres.append((mm[0], qr, qk))
        vres.sort(reverse=True)
        print("       ── VETO top5(val) ──")
        for s, qr, qk in vres[:5]:
            print(f"       rho_r={qr:<5} rho_k={qk:<4} mIoU={s*100:6.2f} ({s*100-b_v[0]*100:+.2f})")
        best["veto"] = (vres[0][1], vres[0][2])
        pwv = []
        for g in [float(x) for x in args.pw_gammas.split(",")]:
            cam = pw_cam_of(mv, Pv, g, args.block)
            mm = evaluate_gated(cam, lab0v, gv, n)
            pwv.append((mm[0], g))
            print(f"       PWF gamma={g:<4} mIoU={mm[0]*100:6.2f} ({mm[0]*100-b_v[0]*100:+.2f})", flush=True)
            del cam
        pwv.sort(reverse=True)
        best["pwf"] = pwv[0][1]
        ops = ["mean", "median", "p25", "trim1"]
        cov = consensus_ops_eval(mv, lab0v, gv, n, ops, block=48)
        print("       ── 逐像素跨成员共识融合(val) ──")
        for op in ops:
            mm = cov[op]
            print(f"       op={op:<7} mIoU={mm[0]*100:6.2f} ({mm[0]*100-b_v[0]*100:+.2f})", flush=True)
        best["cons"] = max(ops, key=lambda o: cov[o][0])
        del mcv, kpv
        val_vote = vote_ensemble(mv, Pv, thr, gv, n, weighted=False)
        print(f"[VAL ] VOTE 多数投票                mIoU={val_vote[0]*100:6.2f} ({val_vote[0]*100-b_v[0]*100:+.2f})", flush=True)
        del val_vote

    # ---------------- TEST ----------------
    print("\n=== TEST ===", flush=True)
    mct = mean_cam(mt, args.block)
    mut, sdt, rt, _, _ = feats(Pt)
    kpt = kappa_of(mt, mct, args.block)
    lab0t = labels_from(mut, thr)
    b_t = evaluate_gated(mct, lab0t, gt, n)
    print(f"[TEST] baseline 集成                mIoU={b_t[0]*100:6.2f}  mDice={b_t[1]*100:6.2f} "
          f"clsIoU=[{b_t[2][0]*100:.1f} {b_t[2][1]*100:.1f} {b_t[2][2]*100:.1f} {b_t[2][3]*100:.1f}]",
          flush=True)

    if args.report in ("val", "both"):
        a, b = best["crw"]
        mm = evaluate_gated(mct, lab0t * (rt ** a) * (kpt ** b), gt, n)
        print(f"[TEST] + CRW (a={a},b={b})           mIoU={mm[0]*100:6.2f} ({mm[0]*100-b_t[0]*100:+.2f}) "
              f"mDice={mm[1]*100:6.2f} clsIoU=[{mm[2][0]*100:.1f} {mm[2][1]*100:.1f} {mm[2][2]*100:.1f} {mm[2][3]*100:.1f}]",
              flush=True)
        for nm, lab in [("CRW 仅 r", lab0t * rt), ("CRW 仅 kappa", lab0t * kpt)]:
            mm = evaluate_gated(mct, lab, gt, n)
            print(f"[TEST]   {nm:14s}                mIoU={mm[0]*100:6.2f} ({mm[0]*100-b_t[0]*100:+.2f})", flush=True)
        qr, qk = best["veto"]
        mm = evaluate_gated(mct, apply_veto(lab0t, rt, kpt, qr, qk, mut), gt, n)
        print(f"[TEST] + VETO (r>={qr},k>={qk})     mIoU={mm[0]*100:6.2f} ({mm[0]*100-b_t[0]*100:+.2f}) "
              f"mDice={mm[1]*100:6.2f} clsIoU=[{mm[2][0]*100:.1f} {mm[2][1]*100:.1f} {mm[2][2]*100:.1f} {mm[2][3]*100:.1f}]",
              flush=True)
        g = best["pwf"]
        cam = pw_cam_of(mt, Pt, g, args.block)
        mm = evaluate_gated(cam, lab0t, gt, n)
        print(f"[TEST] + PWF (gamma={g})             mIoU={mm[0]*100:6.2f} ({mm[0]*100-b_t[0]*100:+.2f}) "
              f"mDice={mm[1]*100:6.2f} clsIoU=[{mm[2][0]*100:.1f} {mm[2][1]*100:.1f} {mm[2][2]*100:.1f} {mm[2][3]*100:.1f}]",
              flush=True)
        del cam
        op = best["cons"]
        mm = consensus_ops_eval(mt, lab0t, gt, n, [op], block=48)[op]
        print(f"[TEST] + 共识融合({op})            mIoU={mm[0]*100:6.2f} ({mm[0]*100-b_t[0]*100:+.2f}) "
              f"mDice={mm[1]*100:6.2f} clsIoU=[{mm[2][0]*100:.1f} {mm[2][1]*100:.1f} {mm[2][2]*100:.1f} {mm[2][3]*100:.1f}]",
              flush=True)
        vt = vote_ensemble(mt, Pt, thr, gt, n, weighted=False)
        print(f"[TEST] VOTE 多数投票                mIoU={vt[0]*100:6.2f} ({vt[0]*100-b_t[0]*100:+.2f}) "
              f"mDice={vt[1]*100:6.2f} clsIoU=[{vt[2][0]*100:.1f} {vt[2][1]*100:.1f} {vt[2][2]*100:.1f} {vt[2][3]*100:.1f}]",
              flush=True)
        del vt, mct


if __name__ == "__main__":
    main()
