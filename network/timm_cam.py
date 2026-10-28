"""通用 timm 骨干（冻结）+ 轻量多尺度适配器 + CAM 头。

目的：为 LUAD 增加**真正不同架构**的集成成员（ConvNeXt / Swin / ResNet 等），
以检验"跨架构共识投票（CADV）"是否在 LUAD 上也能生效（BCSS 上已 +0.38）。

设计对齐 `network/phikon_cam.py` 与 `network/resnet38_cls.py` 的对外契约：
  - `Net(n_class, hfrm_gsr, hfrm_ch, hfrm_kernel).forward(x)` 返回
    (out_56, out_28_1, out_28_2, out_deep, y_deep, cam_56, cam_28_1, cam_28_2, cam_deep, feat_56)
  - `Net_CAM.forward_cam(x)` 返回 (cam_56, cam_28_1, cam_28_2, cam_deep, y_deep)
  - `get_parameter_groups()` 四组：骨干(冻结→空) / 头部 weight / 头部 bias

骨干由环境变量 `TIMM_BACKBONE` 选择（默认 convnext_tiny），可用 `TIMM_UNFREEZE=N` 解冻末 N 段。
"""
import os
import contextlib

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch                                          # noqa: E402
import torch.nn as nn                                 # noqa: E402
import torch.nn.functional as F                       # noqa: E402
import timm                                           # noqa: E402

BACKBONE = os.environ.get("TIMM_BACKBONE", "convnext_tiny")


class Adapter(nn.Module):
    """1×1 投影 + 2× 转置卷积 + BN（把 timm 各 stage 特征升到统一通道与更高分辨率）。"""

    def __init__(self, in_dim, mid=256):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, mid, 1)
        self.up = nn.ConvTranspose2d(mid, mid, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(mid)

    def forward(self, x):
        return F.relu(self.bn(self.up(self.proj(x))))


class Net(nn.Module):
    def __init__(self, n_class, hfrm_gsr=True, hfrm_ch=True, hfrm_kernel=15, mid=256):
        super().__init__()
        self.n_class = n_class
        self.mid = mid
        self.backbone = timm.create_model(BACKBONE, pretrained=True, features_only=True)
        chs = list(self.backbone.feature_info.channels())
        nlev = len(chs)
        # 取最后 4 个 stage（多尺度），不足则重复最后一个
        idx = [max(0, nlev - 4 + i) for i in range(4)]
        self.scale_idx = idx
        self.scale_chs = [chs[i] for i in idx]
        self.adapters = nn.ModuleList([Adapter(chs[i], mid) for i in idx])
        self.heads = nn.ModuleList([nn.Conv2d(mid, n_class, 1) for _ in idx])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.unfreeze_last = int(os.environ.get("TIMM_UNFREEZE", "0"))
        if self.unfreeze_last > 0:
            for mod in list(self.backbone.children())[-self.unfreeze_last:]:
                for p in mod.parameters():
                    p.requires_grad = True
        self.backbone.eval()
        for h in self.heads:
            nn.init.xavier_uniform_(h.weight)
        print(f"[backbone] timm:{BACKBONE}  channels={chs}  use={[chs[i] for i in idx]}", flush=True)

    def _features(self, x):
        self.backbone.eval()
        ctx = torch.no_grad() if self.unfreeze_last == 0 else contextlib.nullcontext()
        with ctx:
            feats = self.backbone(x)
        out = []
        for j, k in enumerate(self.scale_idx):
            f = feats[k]
            if f.shape[1] != self.scale_chs[j]:
                f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out

    def forward(self, x):
        raw = self._features(x)
        cams, outs, feat56 = [], [], None
        for k, (ad, hd, f) in enumerate(zip(self.adapters, self.heads, raw)):
            z = ad(f)
            logit = hd(z)
            cams.append(F.relu(logit))
            outs.append(F.adaptive_avg_pool2d(logit, 1).flatten(1))
            if k == 0:
                feat56 = z
        out_56, out_28_1, out_28_2, out_deep = outs
        y_deep = torch.sigmoid(out_deep)
        return (out_56, out_28_1, out_28_2, out_deep, y_deep,
                cams[0], cams[1], cams[2], cams[3], feat56)

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for m in list(self.adapters) + list(self.heads):
            for name, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                (groups[3] if ("bias" in name) else groups[2]).append(p)
        if self.unfreeze_last > 0:
            for name, p in self.backbone.named_parameters():
                if p.requires_grad:
                    (groups[1] if ("bias" in name) else groups[0]).append(p)
        return groups


class Net_CAM(Net):
    def forward(self, x):
        return super().forward(x)[4]

    def forward_cam(self, x):
        out = super().forward(x)
        return out[5], out[6], out[7], out[8], out[4]
