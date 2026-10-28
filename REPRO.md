# 复现说明（REPRO）

从零复现论文全部主结果、消融与图表的完整步骤。流水线：

```
① 训练成员模型（族 R：ResNet38；族 F：Phikon v1 / v2）
② 采集成员缓存：融合 CAM + 图像级概率（research/cacg_collect.py → /tmp/ens）
③ 池级聚合评估：VOTE / wVOTE / FBV（research/cacg.py）
④ 一键校验（research/verify_repro.py）+ 图表（research/make_figs.py 等）
```

> 所有成员**仅使用图像级标签**训练/适配；像素级标注只用于验证集选择与测试集评估。
> 成员池按验证集选择：BCSS 保留 11 个 R 成员 + 12 个 PFM 成员；LUAD 保留 4 个 R 结构变体 + 2 个 PFM 成员
> （LUAD 的 8 / 15 成员变体属探索性分析，见论文限制说明）。

---

## 0. 环境与前置

```bash
conda create -n xaecadv python=3.10 -y && conda activate xaecadv
pip install -r requirements.txt
pip install mxnet==1.9.1          # 仅 .params → torch 转换需要（一次即可）
mkdir -p init_weights checkpoints checkpoints_ms checkpoints_ablation checkpoints_phikon checkpoints_p2fm

# 若容器内 torch.cuda.is_available() = False（NGC 等），先：
# export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH
# 国内下载 Hugging Face 权重可选：
# export HF_ENDPOINT=https://hf-mirror.com
```

Phikon 权重默认离线加载（`network/phikon_cam.py` 设 `HF_HUB_OFFLINE=1`）。首次运行前预热缓存：

```bash
HF_HUB_OFFLINE=0 python -c "from transformers import AutoModel; AutoModel.from_pretrained('owkin/phikon')"
HF_HUB_OFFLINE=0 python -c "from transformers import AutoModel; AutoModel.from_pretrained('owkin/phikon-v2')"
```

**按 README 的下载链接**放置：
- `init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params`（ResNet38 的 ImageNet 初始化）
- `datasets/BCSS-WSSS/`、`datasets/LUAD-HistoSeg/`（官方发行版原样解压）

---

## 1. 训练成员模型

所有成员共用同一入口 `train_member.py`；两族差别只有 `--network` 与超参。

### 1.1 族 R：ResNet38（端到端，ImageNet 初始化）

模板（以 BCSS seed 11 为例）：

```bash
python train_member.py \
  --dataset bcss --seed 11 --max_epoches 25 --batch_size 20 --lr 0.01 \
  --weights init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --trainroot datasets/BCSS-WSSS/training \
  --save_folder checkpoints_ms/bcss_s11 \
  --eval_every 0 --save-last-k-checkpoints 0 \
  --n_class 4 --img_size 224 --num_workers 8 --amp-dtype bf16
```

BCSS 的 11 个族 R 成员（正式基线 = `bash run_member.sh bcss-train`）：

| tag | seed | 额外参数 | `--save_folder` |
|---|---|---|---|
| `k15_s42` | 42 | 无（正式基线） | `checkpoints/bcss_seed42` |
| `k15_s11` | 11 | 无 | `checkpoints_ms/bcss_s11` |
| `k15_s21` | 21 | 无 | `checkpoints_ms/bcss_s21` |
| `k15_s33` | 33 | 无 | `checkpoints_ms/bcss_s33` |
| `k15_s44` | 44 | 无 | `checkpoints_ms/bcss_s44` |
| `k21_s42` | 42 | `--hfrm_kernel 21` | `checkpoints_ablation/bcss_K21_seed42` |
| `k7_s42` | 42 | `--hfrm_kernel 7` | `checkpoints_ablation/bcss_K7_seed42` |
| `bl_s42` | 42 | `--no-gsr --no-ch` | `checkpoints_ablation/bcss_BL_seed42` |
| `gsr_s42` | 42 | `--no-ch` | `checkpoints_ablation/bcss_GSR_seed42` |
| `ch15_s42` | 42 | `--no-gsr` | `checkpoints_ablation/bcss_CH15_seed42` |
| `aug_s42` | 42 | `--aug` | `checkpoints_aug/bcss_aug` |

LUAD 的族 R 成员（`--dataset luad`，trainroot `datasets/LUAD-HistoSeg/training`）：
`k15_s11` → `checkpoints/luad_seed11`（正式基线 = `bash run_member.sh luad-train`）、
`k15_s42/s21/s33/s44` → `checkpoints_ms/luad_s*`、`k15_s45` → `checkpoints_p2fm/luad_s45`、
`k7/bl/gsr/ch15/k21` → `checkpoints_ablation/luad_<CFG>_seed11`。

> **一键跑 HFRM 结构消融**（BL / GSR / CH15 / K7 / K15 / K21 六配置，自动评估并写 `ablation_results.csv`）：
> ```bash
> bash run_ablation.sh bcss 42
> bash run_ablation.sh luad 11
> ```

### 1.2 族 F：Phikon v1 / Phikon-v2（冻结骨干 + 适配器）

Phikon v1 模板（`--lr 0.001`；骨干由 Hugging Face 加载，`--weights ""` 跳过 ImageNet 加载）：

```bash
PHIKON_ID=owkin/phikon python train_member.py \
  --network network.phikon_cam --dataset bcss --seed 42 --lr 0.001 \
  --max_epoches 25 --batch_size 20 --weights "" \
  --trainroot datasets/BCSS-WSSS/training \
  --save_folder checkpoints_phikon/bcss_seed42 \
  --eval_every 0 --save-last-k-checkpoints 0 \
  --n_class 4 --img_size 224 --num_workers 8 --amp-dtype bf16
```

BCSS 的 12 个族 F 成员：

| tag | 基础模型 | 变体参数 | `--save_folder` |
|---|---|---|---|
| `pv1_seed42` | Phikon v1 | `--seed 42`（默认） | `checkpoints_phikon/bcss_seed42` |
| `pv1_s11` | Phikon v1 | `--seed 11` | `checkpoints_phikon/bcss_s11` |
| `pv1_s2024` | Phikon v1 | `--seed 2024` | `checkpoints_phikon/bcss_s2024` |
| `pv1_e100` | Phikon v1 | `--max_epoches 100` | `checkpoints_phikon/bcss_e100` |
| `pv1_ft4` | Phikon v1 | `PHIKON_UNFREEZE=4 --lr 2e-4`（解冻末 4 个 block） | `checkpoints_phikon/bcss_ft4` |
| `pv1_ft4lr` | Phikon v1 | `PHIKON_UNFREEZE=4 --lr 2e-5` | `checkpoints_phikon/bcss_ft4lr` |
| `pv1_448` | Phikon v1 | `--img_size 448` | `checkpoints_phikon/bcss_448` |
| `pv2` | Phikon-v2 | `--lr 0.01 --seed 42`（`PHIKON_ID=owkin/phikon-v2`） | `checkpoints_p2fm/bcss_pv2` |
| `pv2s11 / s21 / s33 / s44` | Phikon-v2 | 同上，`--seed 11/21/33/44` | `checkpoints_p2fm/bcss_pv2_s11` 等 |

LUAD 的族 F 成员（`--dataset luad`，trainroot `datasets/LUAD-HistoSeg/training`）：
`pv1_seed11` → `checkpoints_phikon/luad_seed11`（v1，`--lr 0.001 --seed 11`）、
`pv2s11` → `checkpoints_p2fm/luad_pv2_s11`（v2，`--lr 0.01 --seed 11`）；
另有 `luad_ft4` / `luad_ft4lr`（`checkpoints_phikon/`，供扩展池实验复现）。

### 1.3 成员 ⇄ 权重路径映射

`research/cacg_collect.py` 顶部两张映射表（`R38` 字典与 `phikon_specs()`）定义每个 tag 对应的
checkpoint 路径与结构配置（GSR/CH/kernel）。若你的训练产物路径与上表不同，
**直接修改该映射表指向你的 `--save_folder`** 即可，其余脚本无需改动。

---

## 2. 采集成员缓存

`research/cacg_collect.py` 逐成员对 val + test 推理一遍，缓存**融合 CAM（fp16）与图像级概率**到 `/tmp/ens/`
（文件名 `<dataset>_<tag>_<split>_fused.npy / _probs.npy`）。缺权重的成员会被自动跳过（`[skip]`）。

```bash
# BCSS：11 + 7 + 5 = 23 成员
python research/cacg_collect.py --dataset bcss --family r38
PHIKON_ID=owkin/phikon    python research/cacg_collect.py --dataset bcss --family pv1
PHIKON_ID=owkin/phikon-v2 python research/cacg_collect.py --dataset bcss --family pv2

# LUAD：11（族 R）+ 3（v1）+ 1（v2）
python research/cacg_collect.py --dataset luad --family r38
PHIKON_ID=owkin/phikon    python research/cacg_collect.py --dataset luad --family pv1
PHIKON_ID=owkin/phikon-v2 python research/cacg_collect.py --dataset luad --family pv2
```

> - 只采部分成员：加 `--tags k15_s42,pv1_seed42`
> - 并行分片采集：按 `--tags` 拆成多条命令（产物互不覆盖）
> - 缓存目录可用 `--cache_dir` 指定，但 `research/cacg.py`（`--dir`）与 `verify_repro.py`（默认 `/tmp/ens`）默认读 `/tmp/ens`
> - `/tmp` 易失，长期保存请转存到项目目录并同步修改默认值

单成员指标（官方口径）也可单独评估：

```bash
python train_member.py --evaluate-only --dataset bcss --seed 42 \
  --weights checkpoints/bcss_seed42/stage1_last.pth \
  --testroot datasets/BCSS-WSSS/test \
  --n_class 4 --img_size 224 --num_workers 8 --amp-dtype bf16
```

> 注意：官方 `--evaluate-only` 与缓存口径（`cacg.py`）存在 ±0.05 的脚本级偏差
> （例：`k21_s42` 官方 70.66 / 缓存 70.62）。集成结果统一使用缓存口径，内部自洽可比。

---

## 3. 复现论文结果表（test mIoU, %）

> 所有池通过 `--tags` 指定；下述命令输出含 **mean（打分平均）/ VOTE / wVOTE / 逐类 IoU**。
> 各表数值已用本仓库缓存口径逐行复核（含补充行 G4-only / G4+PFMs / R11+PFMs(LUAD) / F-only(LUAD)，±0.02 内一致）。

### 3.1 跨架构池消融（论文 Table "Cross-architecture pool ablation with VOTE"）

| Configuration | Pool (BCSS / LUAD) | BCSS | LUAD |
|---|---|---|---|
| Same-arch. | R11 / G4 | 70.76 | 77.90 |
| **Cross-arch.** | R11+12 PFMs / G4+2 PFMs | **72.37** | **78.57** |

```bash
# Same-arch.（同架构池）
python research/cacg.py --dataset bcss --tags aug_s42,bl_s42,ch15_s42,gsr_s42,k15_s11,k15_s21,k15_s33,k15_s42,k15_s44,k21_s42,k7_s42   # → VOTE 70.76
python research/cacg.py --dataset luad --tags bl_s11,ch15_s11,gsr_s11,k7_s11                                                           # → VOTE 77.90

# Cross-arch.（XAE 跨架构池 = 本文方法）
python research/cacg.py --dataset bcss                                                                                                 # 23 成员 → VOTE 72.37
python research/cacg.py --dataset luad --tags bl_s11,ch15_s11,gsr_s11,k7_s11,pv1_seed11,pv2s11                                         # 6 成员 → VOTE 78.57
```

### 3.2 CADV 投票规则消融（论文 Table "CADV voting-rule ablation"）

| Pool | Rule | BCSS | LUAD |
|---|---|---|---|
| Same-arch. | VOTE | 70.76 | 77.90 |
| Cross-arch. | VOTE | 72.37 | **78.57** |
| Cross-arch. | wVOTE | **72.40** | 78.41 |
| Cross-arch. | FBV | 72.31 | 77.24 |

```bash
python research/verify_repro.py     # 每条池输出 mean / VOTE / wVOTE（含 23 成员池与 LUAD 6 成员池）
```

FBV（`member_w = 1/族大小`）内联复现（以 BCSS 23 成员为例；LUAD 把 tags 换成 6 成员、族大小改为 4/2）：

```bash
python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from research.cacg import discover, load_probs, vote_ensemble
mm = [m for m in discover('/tmp/ens', 'bcss', 'test')]
tags = [m[0] for m in mm]
P, g = load_probs([m[2] for m in mm])
thr = np.asarray([0.8, 0.9, 0.8, 0.6], np.float32)
w = [1/12 if t.startswith('pv') else 1/11 for t in tags]   # 族 F 12 成员 / 族 R 11 成员
print('FBV test mIoU = %.2f' % (vote_ensemble(mm, P, thr, g, 4, weighted=False, member_w=w)[0]*100))
"
```

### 3.3 逐类 IoU（论文 Table "Per-class IoU ... same-arch vs cross-arch pools, VOTE"）

| Pool | 数据集 | 逐类 IoU（TUM/TE, STR/TAS, LYM, NEC 顺序） | mIoU |
|---|---|---|---|
| Same-arch. | BCSS | 79.4 / 73.4 / **60.8** / 69.5 | 70.76 |
| Cross-arch. | BCSS | **82.4** / **75.6** / 60.6 / **70.9** | **72.37** |
| Same-arch. | LUAD | 77.4 / 83.3 / 77.5 / 73.4 | 77.90 |
| Cross-arch. | LUAD | **78.3** / **84.4** / **77.6** / **74.0** | **78.57** |

逐类 IoU 由 §3.1 的同一批命令直接打印（`cacg.py` 输出 `clsIoU=[...]`）。

### 3.4 成员池结构消融（论文 Table "Member-pool structure ablation with VOTE"）

| Pool | Composition | BCSS | LUAD |
|---|---|---|---|
| G4-only | 4 个 R38 结构变体 | 69.77 | 77.90 |
| R11-only | 11 个 R38 成员 | 70.76 | 77.54 |
| G4 + PFMs | G4 + 2 PFMs（M=6） | 70.83 | **78.57** |
| R11 + PFMs | BCSS: R11+12 PFMs；LUAD: R11+2 PFMs | **72.37** | 77.89 |

```bash
# G4-only（BCSS 用同定义的 4 个结构变体标签）
python research/cacg.py --dataset bcss --tags bl_s42,ch15_s42,gsr_s42,k7_s42
python research/cacg.py --dataset luad --tags bl_s11,ch15_s11,gsr_s11,k7_s11
# G4 + 2 PFMs
python research/cacg.py --dataset bcss --tags bl_s42,ch15_s42,gsr_s42,k7_s42,pv1_s11,pv2s11
python research/cacg.py --dataset luad --tags bl_s11,ch15_s11,gsr_s11,k7_s11,pv1_seed11,pv2s11
# R11 + PFMs（LUAD：11×R38 + 2 PFMs）
python research/cacg.py --dataset luad --tags k15_s11,k15_s42,k15_s21,k15_s33,k15_s44,bl_s11,gsr_s11,ch15_s11,k7_s11,k21_s11,k15_s45,pv1_seed11,pv2s11
```

### 3.5 架构多样性贡献（论文 Table "VOTE by architecture-family composition"）

| Pool | Members (BCSS/LUAD) | BCSS | LUAD |
|---|---|---|---|
| R family only | 11 / 4 | 70.76 | 77.90 |
| F family only | 12 / 2 | 71.04 | 66.98 |
| **R + F** | **23 / 6** | **72.37** | **78.57** |

```bash
# 仅族 F（12 个 Phikon v1/v2）
python research/cacg.py --dataset bcss --tags pv1_448,pv1_e100,pv1_ft4,pv1_ft4lr,pv1_s11,pv1_s2024,pv1_seed42,pv2,pv2s11,pv2s21,pv2s33,pv2s44
# 仅族 F（2 个：v1+v2）
python research/cacg.py --dataset luad --tags pv1_seed11,pv2s11
```

### 3.6 族比例消融（BCSS，论文 Table "Family-proportion ablation"）

| Pool | M | VOTE | FBV |
|---|---|---|---|
| R11 + 2 PFMs | 13 | 71.02 | **72.04** |
| R11 + 6 PFMs | 17 | 71.58 | **72.03** |
| R11 + 12 PFMs | 23 | **72.37** | 72.31 |

```bash
python research/cacg.py --dataset bcss --tags aug_s42,bl_s42,ch15_s42,gsr_s42,k15_s11,k15_s21,k15_s33,k15_s42,k15_s44,k21_s42,k7_s42,pv1_s11,pv2s11
python research/cacg.py --dataset bcss --tags aug_s42,bl_s42,ch15_s42,gsr_s42,k15_s11,k15_s21,k15_s33,k15_s42,k15_s44,k21_s42,k7_s42,pv1_s11,pv2s11,pv1_448,pv1_e100,pv1_ft4,pv1_ft4lr
# FBV 见 §3.2 内联脚本（把族大小权重改为 1/11 与 1/#PFM）
```

### 3.7 成员决策一致率（论文 Table "pairwise agreement"，探索性）

```bash
python research/make_figs.py       # fig6 打印 intra-R / cross-family / intra-F 平均像素一致率
```

---

## 4. HFRM 结构消融（单成员）

```bash
bash run_ablation.sh bcss 42       # BL / GSR / CH15 / K7 / K15 / K21
bash run_ablation.sh luad 11
```

产物：`checkpoints_ablation/<ds>_<config>_seed<seed>/stage1_last.pth`、`logs/ablation_*.log`、
汇总 `ablation_results.csv`（字段：dataset / config / seed / mIoU / mDice / checkpoint）。
本仓库已附实测汇总：`results/ablation_results.csv`。

---

## 5. 复现图

```bash
# 补充图：架构示意 / 决策共识示意 / 定性对比 / 族比例增益 / 逐类 IoU / 成员一致率 / 累积消融
python research/make_figs.py         # 产物 research/figs/*.png；并生成 .maps_bcss_test.npz

# 定性结果网格图（LUAD 3 行 + BCSS 3 行，5 列，含 ROI 白框）
python research/make_luad_maps.py    # 生成 .maps_luad_test.npz（LUAD：单模型 / mean / VOTE / wVOTE 预测）
python research/make_fig3_grid.py    # 产物 research/figs/fig3_qualitative_v8.png
```

> `make_fig3_grid.py` 依赖 `/tmp/ens` 缓存（第 2 节产物）与两个 `.maps_*.npz`；
> 图像顺序严格采用数据集官方 `os.walk` 顺序（与 `Stage1_InferDataset` 一致），**不要改成 sorted(glob)**，否则索引错位。

---

## 6. 门控与族级 FP/TP 保留率分析（论文 Discussion）

### 6.1 门控缺陷分解（只去假阳 vs 只补假阴）

```bash
# 先导出单成员缓存 npz（含 gts），再分解
python research/softgate_sweep.py --dataset bcss --weights checkpoints/bcss_seed42/stage1_last.pth \
    --testroot datasets/BCSS-WSSS/val --cache /tmp/sg_bcss_val_seed42.npz
python research/gate_decomp.py   --cache /tmp/sg_bcss_val_seed42.npz --dataset bcss
python research/gate_fp_stats.py --cache /tmp/sg_bcss_val_seed42.npz --dataset bcss
```

历史结论（BCSS val）：只去假阳（232 个实例）价值 **+2.28 mIoU**，只补假阴（1521 个实例）价值 **+0.75 mIoU**。

### 6.2 族级 FP/TP 保留率（`research/make_figs.py::_delta_of`）

按论文定义：为每个 **(验证图像, 目标类)** 构造一个实例。

- $z_{i,k}=1$：类 $k$ 在图像 $i$ 的像素标注中确实存在；
- $q_{i,k}=\mathbb{I}[\frac{1}{M}\sum_m p^{(m)}_{i,k}>\tau_k]$：全池成员平均概率给出的类存在性预测；
- **TP 实例** = $z=1 \wedge q=1$；**FP 实例** = $z=0 \wedge q=1$（**分母分别为 TP / FP 实例数，不是像素数**）；
- 族级存在性：族内成员平均概率过阈；两族**严格多数**（两族时需两族均为正）得到共识指示 $b_{i,k}$；
- $\rho_{\mathrm{TP}}=\mathbb{E}[b\mid z{=}1,q{=}1]$，$\rho_{\mathrm{FP}}=\mathbb{E}[b\mid z{=}0,q{=}1]$，$\Delta=\rho_{\mathrm{TP}}-\rho_{\mathrm{FP}}$。

实测（val，描述性统计）：BCSS ρ_FP 0.68 / ρ_TP 0.94 / **Δ 0.26**；LUAD 0.90 / 0.99 / **Δ 0.09**。
论文中该统计量仅作**描述性诊断**，不作为投票算子的选择规则。

---

## 7. 复现对照表（±0.02 容差）

| 数据集 | split | 池 | mean | VOTE | wVOTE |
|---|---|---|---|---|---|
| BCSS | val | 23 成员 | 70.14 | 70.43 | 70.24 |
| BCSS | test | 23 成员 | 71.99 | 72.37 | 72.40 |
| BCSS | test | 同架构 R11 | 70.72 | 70.76 | — |
| LUAD | val | 4 变体 + 2 PFM | 77.07 | 77.16 | — |
| LUAD | test | 4 变体 + 2 PFM | 79.04 | 78.57 | — |
| LUAD | test | 4 变体 | 78.13 | 77.90 | — |
| LUAD | test | R11 | 77.83 | 77.54 | — |

---

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `torch.cuda.is_available() = False` | `export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH` |
| Phikon 加载报 "offline mode" | 预热 HF 缓存，或 `export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0`；国内加 `HF_ENDPOINT=https://hf-mirror.com` |
| `/tmp/ens` 被系统清理 | 重新运行 `cacg_collect.py`（可用 `--tags` 分片）；长期保存请转存并修改默认路径 |
| 结果与论文差 ~0.05 | 官方 `--evaluate-only` 与缓存口径的脚本级偏差，集成结果统一用缓存口径 |
| collect 时显存不足 | 逐成员执行（`--tags` 单个成员），或调小并发；缓存为 fp16 + mmap 读取，评估阶段无需 GPU |
| `mxnet` 安装失败 | 仅 `.params` 转换需要；可先在本机把 `.params` 转成 `.pth` 后带过来用（`--weights xxx.pth`） |