"""Phikon（病理基础模型，ViT-B/16，4000 万张 pan-cancer WSI 预训练）作为冻结特征提取器，
接轻量多尺度适配器与 CAM 头，保持与 train_member.py / tool/infer_fun.py 完全兼容的接口。

设计（对应"换骨干 + 引入基础模型先验"）：
  - backbone：`owkin/phikon` 冻结（requires_grad=False），12 层 ViT-B/16，224 输入 -> 14x14 patch token
  - 多尺度：取第 12 层的 4 个中间层输出 (3, 6, 9, 12)，去掉 CLS token 后 reshape 成 14x14 特征图
  - 适配器：1x1 投影(768->256) + 可学习 2x 转置卷积 + BN -> 28x28，缓解 ViT token 分辨率过粗的问题
  - CAM 头：每尺度 1x1 卷积到 n_class；out_s = GAP(logits)，cam_s = ReLU(logits)，y_deep = sigmoid(out_deep)
  - 训练目标、推理路径、多尺度融合与 ResNet38 版本完全一致（便于与原方法严格对照）

注意：本模块**不覆写** train()/eval()，以免破坏 nn.Module.eval() 的链式调用。
"""
import os
import contextlib

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch                                          # noqa: E402
import torch.nn as nn                                 # noqa: E402
import torch.nn.functional as F                       # noqa: E402
from transformers import AutoModel                    # noqa: E402

PHIKON_ID = os.environ.get("PHIKON_ID", "owkin/phikon")   # 可为 owkin/phikon-v2（ViT-L/24 层）
PATCH_GRID = 14                                       # 224 / 16
SCALE_LAYERS = (3, 6, 9, 12)                          # 默认层级（按骨干深度自动推导）


def _default_layers(depth):
    env = os.environ.get("PHIKON_LAYERS")
    if env:
        return tuple(int(x) for x in env.split(","))
    if depth == 12:
        return (3, 6, 9, 12)
    return tuple(max(1, round(depth * r)) for r in (0.25, 0.5, 0.75, 1.0))


class PhikonAdapter(nn.Module):
    """把某一层的 ViT token 特征投影并上采样到 2x 空间分辨率，作为 CAM 的载体。"""

    def __init__(self, in_dim=768, mid=256):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, mid, 1)
        self.up = nn.ConvTranspose2d(mid, mid, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(mid)

    def forward(self, x):                             # x: [B, in_dim, 14, 14]
        x = self.proj(x)
        x = self.bn(self.up(x))                       # -> [B, mid, 28, 28]
        return F.relu(x)


class Net(nn.Module):
    def __init__(self, n_class, hfrm_gsr=True, hfrm_ch=True, hfrm_kernel=15, mid=256):
        super().__init__()
        self.n_class = n_class
        self.mid = mid
        try:
            # Phikon (ViTModel) 支持 add_pooling_layer；Phikon-v2 (Dinov2Model) 不支持
            self.backbone = AutoModel.from_pretrained(PHIKON_ID, add_pooling_layer=False)
        except TypeError:
            self.backbone = AutoModel.from_pretrained(PHIKON_ID)
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 可选微调：解冻最后 N 层编码器（环境变量 PHIKON_UNFREEZE=N，默认 0 = 全冻结）
        self.unfreeze_last = int(os.environ.get("PHIKON_UNFREEZE", "0"))
        if self.unfreeze_last > 0:
            for layer in self.backbone.encoder.layer[-self.unfreeze_last:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True
        self.backbone.eval()

        self.hidden = int(self.backbone.config.hidden_size)
        self.scale_layers = _default_layers(int(self.backbone.config.num_hidden_layers))
        print(f"[backbone] {PHIKON_ID}  hidden={self.hidden}  layers={self.scale_layers}", flush=True)

        self.adapters = nn.ModuleList([PhikonAdapter(self.hidden, mid) for _ in self.scale_layers])
        self.heads = nn.ModuleList([nn.Conv2d(mid, n_class, 1) for _ in self.scale_layers])
        for h in self.heads:
            nn.init.xavier_uniform_(h.weight)

    def _features(self, x):
        self.backbone.eval()                          # 冻结骨干始终 eval（避免 BN/Dropout 漂移）
        side = x.shape[-1]
        grid = side // self.backbone.config.patch_size
        interp = side != self.backbone.config.image_size       # 非 224 输入需插值位置编码
        # 全冻结时用 no_grad 节省显存；微调时需保留计算图
        ctx = torch.no_grad() if self.unfreeze_last == 0 else contextlib.nullcontext()
        with ctx:
            # Dinov2Model 的 forward 不接受 interpolate_pos_encoding；224 输入时无需插值
            kw = {"interpolate_pos_encoding": True} if interp else {}
            hs = self.backbone(x, output_hidden_states=True, **kw).hidden_states
        feats = []
        for li in self.scale_layers:
            h = hs[li][:, 1:, :]                      # 去掉 CLS token -> [B,grid*grid,C]
            b, n, c = h.shape
            feats.append(h.transpose(1, 2).reshape(b, c, grid, grid))
        return feats

    def forward(self, x):
        raw = self._features(x)
        feats, cams, outs = [], [], []
        for ad, hd, f in zip(self.adapters, self.heads, raw):
            z = ad(f)
            logit = hd(z)
            feats.append(z)
            cams.append(F.relu(logit))
            outs.append(F.adaptive_avg_pool2d(logit, 1).flatten(1))
        out_56, out_28_1, out_28_2, out_deep = outs
        y_deep = torch.sigmoid(out_deep)
        return (out_56, out_28_1, out_28_2, out_deep, y_deep,
                cams[0], cams[1], cams[2], cams[3], feats[0])

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for m in list(self.adapters) + list(self.heads):
            for name, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                (groups[3] if ("bias" in name) else groups[2]).append(p)
        if self.unfreeze_last > 0:          # 解冻的骨干层走 group0/1（小学习率）
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
