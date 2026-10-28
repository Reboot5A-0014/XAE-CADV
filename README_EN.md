# XAE-CADV

**XAE-CADV: Cross-Architecture Ensemble and Decision Voting for Weakly Supervised Histopathology Image Segmentation**

Official code release (IEEE Journal of Biomedical and Health Informatics). ｜ [中文文档](README.md)

## Method Overview

XAE-CADV is an **inference-stage** aggregation framework: it leaves each member's training objective, network architecture, and forward computation unchanged, and improves pixel-level predictions with two plug-in components.

| Component | Role | Description |
|---|---|---|
| **Module I · XAE**<br>Cross-Architecture Ensemble (Eq. 1) | Cross-architecture member pool | $\mathcal{M}=\mathcal{M}_R\cup\mathcal{M}_F$: family $\mathcal{M}_R$ contains ResNet38 CNN members; family $\mathcal{M}_F$ contains Phikon v1 / Phikon-v2 pathology foundation models (frozen backbone + lightweight adapters). All members are trained/adapted with **image-level labels only**. |
| **Module II · CADV**<br>Cross-Architecture Decision Voting (Eq. 7–14) | Pixel-wise decision voting | Each member first produces an independent discrete pixel prediction (Eq. 6); class votes are accumulated per pixel (Eq. 7) and reduced with argmax (Eq. 8). Three parameter-free rules: **VOTE** (Eq. 9–10), **wVOTE** (Eq. 11–12), **FBV** (Eq. 13–14). CADV adds no training and no forward pass beyond the member predictions. |

![framework](assets/framework.png)

**Shared per-member inference pipeline** (Eq. 2–6): 3× TTA (original / horizontal flip / vertical flip) → CAMs are resized to the original grid and inverse-transformed, then **averaged across augmentations before** per-class min–max normalization → multi-scale fusion with fixed weights `0.6/0.2/0.2` → image-level hard-threshold gating (with an all-zero fallback; BCSS per-class `[0.8,0.9,0.8,0.6]`, LUAD uniform `0.2`) → argmax over active classes.

PFM members do not use raw transformer attention maps. For a 224×224 input the ViT produces a 14×14 patch-token grid; four hidden states (layers {3,6,9,12} for Phikon v1, {6,12,18,24} for Phikon-v2, i.e. ≈25/50/75/100% depth) pass through learnable adapters (1×1 projection → learnable 2× transposed convolution → BN → ReLU) to 28×28, followed by 1×1 CAM heads with the same interface as the CNN members. The first branch serves as an auxiliary branch; the remaining three participate in the multi-scale fusion. Fully frozen configurations train ≈ **5.25 M** parameters.

## Main Results (test mIoU, %)

| Configuration (within the same member pool) | BCSS-WSSS | LUAD-HistoSeg |
|---|---|---|
| Same-architecture pool + VOTE | 70.76 | 77.90 |
| **Cross-architecture pool (XAE) + CADV (VOTE)** | **72.37** | **78.57** |

Voting rules on the cross-architecture pool (BCSS / LUAD): wVOTE 72.40 / 78.41, FBV 72.31 / 77.24.

![qualitative](assets/qualitative_results.png)

> The qualitative panels show BCSS-WSSS test patches with visualization colors (TUM red / STR green / LYM blue / NEC yellow); light-gray pixels are ignore regions excluded from metrics. The pixel-accuracy values printed beneath the prediction panels are per-patch descriptors only — the reported metrics use accumulated confusion counts over the complete evaluation split.

## Environment (as in the paper)

- Single **NVIDIA L20 (48 GB)** GPU; PyTorch in an NGC container
- `transformers 4.40.2`, `timm 1.0.30`; Python 3.10

```bash
conda create -n xaecadv python=3.10 -y
conda activate xaecadv
pip install -r requirements.txt
pip install mxnet==1.9.1          # only to convert the ImageNet .params weights to torch (once)
```

## Downloads (external links)

| File | Destination | Link |
|---|---|---|
| ImageNet initialization weights `ilsvrc-cls_rna-a1_cls1000_ep-0001.params` | `init_weights/` | [Google Drive](https://drive.google.com/file/d/1Rka2SzqAwxUEFb28tbmiy2anhkkFOnTg/view?usp=drive_link) |
| BCSS-WSSS dataset | `datasets/BCSS-WSSS/` | [Google Drive](https://drive.google.com/file/d/178eSM9xs5jITt5P2kjaswDlJzwlU5gps/view?usp=drive_link) |
| LUAD-HistoSeg dataset | `datasets/LUAD-HistoSeg/` | [Google Drive](https://drive.google.com/file/d/1lWAeCp6UN30VRVmqv97kA2sJ1Pp2frhC/view?usp=drive_link) |
| Phikon v1 (ViT-B/16) | Hugging Face cache (automatic) | [huggingface.co/owkin/phikon](https://huggingface.co/owkin/phikon) |
| Phikon-v2 (DINOv2 ViT-L/24) | Hugging Face cache (automatic) | [huggingface.co/owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2) |

Dataset layout (just unpack the official releases):

```text
datasets/
├── BCSS-WSSS/{training, val/{img,mask}, test/{img,mask}}/
└── LUAD-HistoSeg/{training, val/{img,mask}, test/{img,mask}}/
```

> Phikon weights are loaded in **offline mode** by default (`network/phikon_cam.py` sets `HF_HUB_OFFLINE=1`). Before the first run, either
> 1) warm the cache: `HF_HUB_OFFLINE=0 python -c "from transformers import AutoModel; AutoModel.from_pretrained('owkin/phikon')"` (same for v2), or
> 2) override at runtime: `export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0` (optionally with `export HF_ENDPOINT=https://hf-mirror.com`).

## Repository Structure

```text
XAE-CADV/
├── train_member.py          # member training / single-member evaluation (--network selects ResNet38 or Phikon)
├── run_member.sh            # convenience script for training and evaluation
├── run_ablation.sh          # one-shot HFRM ablation (BL / GSR / CH15 / K7 / K15 / K21)
├── network/
│   ├── resnet38_cls.py      # family R members: ResNet38 + HFRM + multi-scale CAM branches
│   ├── resnet38d.py         # ResNet38 definition + mxnet→torch weight conversion
│   ├── phikon_cam.py        # family F members: frozen Phikon v1/v2 backbone + adapters + CAM heads
│   └── timm_cam.py          # optional: timm backbone adapter (for extensions)
├── tool/                    # datasets / inference / evaluation / optimizers
│   ├── GenDataset.py        # training and inference datasets (official sample order)
│   ├── infer_fun.py         # inference pipeline (TTA + multi-scale fusion + gating)
│   ├── iouutils.py          # official IoU / mIoU protocol
│   └── ...
├── research/                # reproduction and analysis scripts
│   ├── cacg_collect.py      # cache member outputs (fused CAMs + image-level probabilities) → /tmp/ens
│   ├── cacg.py              # pool-level evaluation: score averaging / VOTE / wVOTE / FBV
│   ├── verify_repro.py      # one-shot reproduction check (compares with paper values, ±0.02)
│   ├── softgate_sweep.py    # single-member TTA inference and gate sweep (can dump npz caches)
│   ├── ens_greedy_val.py    # validation-based member-pool selection
│   ├── gate_decomp.py       # gate-error decomposition (remove FPs vs. recover FNs)
│   ├── gate_fp_stats.py     # separability diagnostics for FP instances (TP/FP feature AUC)
│   ├── make_figs.py         # supplementary figures + family-level FP/TP retention (_delta_of)
│   ├── make_luad_maps.py    # LUAD prediction caches (for qualitative figures)
│   ├── make_fig3_grid.py    # qualitative result grid
│   └── phikon_tune.py       # Phikon inference hyper-parameter diagnostics
├── core_method/core.py      # condensed core implementation (matches the paper equations)
├── assets/                  # figures and figure-drawing script
└── results/ablation_results.csv   # HFRM ablation summary (measured in this repo)
```

## Quick Start (minimal reproduction: train → cache → vote)

```bash
# 1) Train a ResNet38 member (family R, ImageNet init, end-to-end)
bash run_member.sh bcss-train

# 2) Train PFM members (family F, frozen backbone + adapters; --weights "" skips ImageNet loading)
PHIKON_ID=owkin/phikon    python train_member.py --network network.phikon_cam \
  --dataset bcss --seed 42 --lr 0.001 --weights "" \
  --trainroot datasets/BCSS-WSSS/training --save_folder checkpoints_phikon/bcss_seed42 \
  --n_class 4 --img_size 224 --num_workers 8 --amp-dtype bf16 --max_epoches 25
PHIKON_ID=owkin/phikon-v2 python train_member.py --network network.phikon_cam \
  --dataset bcss --seed 42 --lr 0.01  --weights "" \
  --trainroot datasets/BCSS-WSSS/training --save_folder checkpoints_p2fm/bcss_pv2 \
  --n_class 4 --img_size 224 --num_workers 8 --amp-dtype bf16 --max_epoches 25

# 3) Cache member outputs (fused CAMs + image-level probabilities → /tmp/ens)
python research/cacg_collect.py --dataset bcss --family r38 --tags k15_s42
PHIKON_ID=owkin/phikon    python research/cacg_collect.py --dataset bcss --family pv1 --tags pv1_seed42
PHIKON_ID=owkin/phikon-v2 python research/cacg_collect.py --dataset bcss --family pv2 --tags pv2

# 4) Pool-level evaluation (VOTE / wVOTE / score averaging)
python research/cacg.py --dataset bcss --tags k15_s42,pv1_seed42,pv2
```

Full reproduction (23 / 6-member pools, all ablations, figures) is described in **[REPRO.md](REPRO.md)** (Chinese).

## Reproduction Check

`research/verify_repro.py` recomputes score averaging, VOTE, and wVOTE for the key pools and compares
them with the paper values (±0.02 tolerance):

```bash
python research/verify_repro.py
```

| Dataset / pool | Score averaging | VOTE | wVOTE |
|---|---|---|---|
| BCSS · 23 members | 71.99 | **72.37** | 72.40 |
| BCSS · same-architecture R11 | 70.72 | 70.76 | — |
| LUAD · 4×R38 variants + 2 PFMs (cross-architecture, M=6) | 79.04 | **78.57** | 78.41 |
| LUAD · 4×R38 variants | 78.13 | 77.90 | — |
| LUAD · R11 (same-architecture) | 77.83 | 77.54 | — |

## Known Limitations (consistent with the paper)

- **LUAD-HistoSeg split**: the official split is patch-level; validation and test share 14 WSIs (192/300 validation and 181/307 test patches lie on shared slides). This overlap originates from the released split; results should be read as performance under the released protocol rather than slide-disjoint generalization.
- **Exploratory LUAD test result**: the 8- and 15-member LUAD pool variants were inspected on the nominal test split, so the LUAD test estimate is not fully untouched and needs confirmation on a new WSI-disjoint holdout.
- **Pretraining-source overlap**: BCSS-WSSS is derived from TCGA-BRCA, so source-level overlap with the TCGA-derived pretraining corpora of the Phikon family cannot be excluded. The pretraining is self-supervised and uses no pixel labels; per-tile pretraining manifests are not public, so exact overlap cannot be checked.

## Citation

```bibtex
@article{,
  title   = {{XAE-CADV}: Cross-Architecture Ensemble and Decision Voting for
             Weakly Supervised Histopathology Image Segmentation},
  author  = {},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review}
}
```

## Acknowledgments

- Pathology foundation model weights: [Phikon](https://huggingface.co/owkin/phikon) / [Phikon-v2](https://huggingface.co/owkin/phikon-v2) (Owkin)
- Datasets: BCSS-WSSS (derived from BCSS, Amgad et al., *Bioinformatics* 2019), LUAD-HistoSeg

This code is released for academic research use only; third-party datasets and pretrained weights follow their respective licenses.