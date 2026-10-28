import os
import math
import numpy as np
import argparse
import importlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tool import pyutils, torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_fun import infer
import time
import random
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

cudnn.enabled = True
time_test = 0

# =========================================================================
# 0. PALETTE & UTILS
# =========================================================================
LUAD_PALETTE = [
    205, 51, 51,   # Class 0: Tumor
    0, 255, 0,     # Class 1: Stroma
    65, 105, 225,  # Class 2: Normal
    255, 165, 0,   # Class 3: Necrosis
    255, 255, 255  # Class 4: Background
]
LUAD_PALETTE += [0] * (256 * 3 - len(LUAD_PALETTE))

BCSS_PALETTE = [
    255, 0, 0,     # Class 0
    0, 255, 0,     # Class 1
    0, 0, 255,     # Class 2
    153, 0, 255,   # Class 3
    255, 255, 255  # Class 4: Background
]
BCSS_PALETTE += [0] * (256 * 3 - len(BCSS_PALETTE))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    try:
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
    except AttributeError:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def get_amp_dtype(args):
    if args.amp_dtype == 'bf16':
        return torch.bfloat16
    if args.amp_dtype == 'fp16':
        return torch.float16
    return None

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_checkpoint_path(args):
    return os.path.join(args.save_folder, args.checkpoint_name)

def get_infer_thr(args):
    return args.infer_thr if args.infer_thr is not None else None

def get_cam_weights(args):
    return (args.cam_w_28_1, args.cam_w_28_2, args.cam_w_deep)

def get_loss_weights(args):
    return (args.loss_w_56, args.loss_w_28_1, args.loss_w_28_2, args.loss_w_deep)

def apply_palette(mask_np, dataset='luad'):
    mask_img = Image.fromarray(mask_np.astype(np.uint8))
    if dataset == 'bcss':
        mask_img.putpalette(BCSS_PALETTE)
    else:
        mask_img.putpalette(LUAD_PALETTE)
    return mask_img.convert('RGB')

def overlay_heatmap(img_np, cam_np):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

def norm_np(cam_np):
    c_min = np.min(cam_np, axis=(1, 2), keepdims=True)
    c_max = np.max(cam_np, axis=(1, 2), keepdims=True)
    return (cam_np - c_min) / (c_max - c_min + 1e-8)

def spatial_normalize(cam_tensor):
    B_sz, C_sz, H_sz, W_sz = cam_tensor.shape
    cam_flat = cam_tensor.view(B_sz, C_sz, -1)
    c_min = cam_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    c_max = cam_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    denominator = (c_max - c_min).clamp_min(1e-5)
    return (cam_tensor - c_min) / denominator


def aer_loss(cams, label, q=0.3):
    """AER：对抗擦除正则（Adversarial Erasing Regularization）。

    动机（本项目实测诊断）：BCSS 上 CAM 的主要缺陷是**覆盖不足**（LYM 召回 0.61、NEC 0.52、面积比 0.79），
    而任何以"CAM 派生伪掩码"为监督的模块都无法突破 CAM 自身的覆盖范围（已实验证伪）。
    AER 只用**图像级标签**：擦除每个尺度 CAM 中该类最激活的 q 比例像素后，
    仍要求分类正确（等价于把被擦除特征图送回头部再算一次 GAP）——
    这迫使网络在被遮蔽区域之外保留互补证据，从而**扩张**有效激活区域。

    由于头部是 1x1 卷积，擦除特征图 ≡ 擦除 CAM，故本损失无需额外前向，仅一次掩码求和。
    不引入新参数、不改变推理路径。

    cams: [cam_56, cam_28_1, cam_28_2, cam_deep]，ReLU 后 [B,C,h,w]
    """
    total = cams[0].new_zeros(())
    for cam in cams:
        B, C, H, W = cam.shape
        flat = cam.view(B, C, -1)
        k = max(1, int(q * H * W))
        thr = flat.topk(k, dim=-1).values[:, :, -1:]
        keep = (flat < thr).float()                       # 保留"非峰值"区域（擦除峰值 q 比例）
        erased_logit = (flat * keep).sum(-1) / float(H * W)   # 与 GAP 同尺度
        total = total + F.binary_cross_entropy_with_logits(erased_logit, label)
    return total / len(cams)


def pmst_loss(cams, label, thr_lo=0.3, thr_hi=0.7, temp=1.0):
    """PMST：单阶段跨尺度共识扩张自训练（Cross-Scale Consensus Expansion）。

    针对"CAM 只覆盖判别性区域 / 激活不完整"：把各尺度归一化 CAM 的**并集**
    （任一尺度高置信的像素）作为伪掩码，去监督**每一个**尺度自身，
    使各尺度向并集扩张；用软权重（置信度介于 thr_lo~thr_hi 之间线性加权）抑制噪声。
    伪掩码 stop-gradient，避免平凡解。不引入新参数，不改变推理路径。

    cams: [cam_56, cam_28_1, cam_28_2, cam_deep]，均为 ReLU 后的 [B,C,h,w]
    label: [B,C] 图像级标签
    """
    size = cams[0].shape[-2:]
    ups = [F.interpolate(c, size=size, mode="bilinear", align_corners=False) for c in cams]
    with torch.no_grad():
        maps = []
        for u in ups:
            mn = u.amin(dim=(2, 3), keepdim=True)
            mx = u.amax(dim=(2, 3), keepdim=True)
            maps.append((u - mn) / (mx - mn + 1e-5))
        stack = torch.stack(maps, 0)                              # [S,B,C,H,W]
        any_conf = stack.amax(0) * label[:, :, None, None]        # 跨尺度并集，屏蔽未出现的类
        top1, pseudo = any_conf.max(dim=1)                        # [B,H,W]
        w = ((top1 - thr_lo) / (thr_hi - thr_lo)).clamp(0.0, 1.0)
        denom = w.sum().clamp(min=1.0)

    total = cams[0].new_zeros(())
    for u in ups:
        ce = F.cross_entropy(u / temp, pseudo, reduction="none")
        total = total + (ce * w).sum() / denom
    return total / len(ups)


def asym_cls_loss(logits, target, neg_weight=1.0):
    """非对称存在性损失：给"负类"（图像中不存在的类）更高的权重。

    动机（本项目实测机理，research/gate_decomp.py）：BCSS val 上门控缺陷分解显示
      - 只去假阳（232 个实例）→ mIoU +2.28
      - 只补假阴（1521 个实例）→ mIoU +0.75
    即**单个假阳的杀伤力约为假阴的 13 倍**；而原始的 `multilabel_soft_margin_loss`
    对正负类一视同仁，优化目标与下游分割收益并不一致。

    这里把负类项乘以 neg_weight（>1 表示更用力压低"高置信度假阳"）。
    neg_weight == 1.0 时与原始 multilabel_soft_margin_loss 完全等价（已做数值对齐）。
    """
    log_p = F.logsigmoid(logits)
    log_1mp = F.logsigmoid(-logits)
    loss = -(target * log_p + neg_weight * (1.0 - target) * log_1mp)
    return loss.mean()


class ClassificationEvalDataset(Dataset):
    def __init__(self, data_root, transform=None, img_size=224, n_class=4, dino_version="dino_v2"):
        self.img_dir = os.path.join(data_root, 'img')
        self.mask_dir = os.path.join(data_root, 'mask')
        self.bg_mask_dir = os.path.join(os.path.dirname(data_root.rstrip('/')), 'bg_mask_test')
        self.dino_feat_dir = os.path.join(os.path.dirname(data_root.rstrip('/')), f'test_feats_{dino_version}_{img_size}')
        
        self.img_size = img_size
        self.n_class = n_class
        self.transform = transform 
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if not f.startswith('.')]

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(os.path.join(self.img_dir, img_id + '.png')).convert('RGB')
        
        mask_np = np.array(Image.open(os.path.join(self.mask_dir, img_id + '.png')))
        
        bg_mask = Image.open(os.path.join(self.bg_mask_dir, img_id + '.png')).convert('L') if os.path.exists(os.path.join(self.bg_mask_dir, img_id + '.png')) else Image.new('L', (self.img_size, self.img_size), 0)
        
        feat_path = os.path.join(self.dino_feat_dir, img_id + '.pt')
        dino_feat = torch.load(feat_path) if os.path.exists(feat_path) else torch.zeros((196, 384))
        if len(dino_feat.shape) == 3: dino_feat = dino_feat.view(-1, dino_feat.shape[-1])

        if img.size[0] != self.img_size or img.size[1] != self.img_size:
            img = TF.resize(img, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
            bg_mask = TF.resize(bg_mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
            
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        bg_mask_tensor = (torch.from_numpy(np.array(bg_mask)).long() > 128).float()
        
        label_tensor = torch.zeros(self.n_class)
        for c in np.unique(mask_np):
            if c < self.n_class: label_tensor[c] = 1.0

        return img, label_tensor, bg_mask_tensor, dino_feat

    def __len__(self): return len(self.ids)

def compute_acc(pred_labels, gt_labels):
    pred_correct_count = len(set(pred_labels) & set(gt_labels))
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    return round(pred_correct_count / union, 4) if union > 0 else 1.0


def train_phase(args):
    global time_test

    set_seed(args.seed)
    model = getattr(importlib.import_module(args.network), 'Net')(
        n_class=args.n_class, hfrm_gsr=args.hfrm_gsr, hfrm_ch=args.hfrm_ch,
        hfrm_kernel=args.hfrm_kernel).cuda()
    print('[Ablation] gsr=%s ch=%s kernel=%d' % (args.hfrm_gsr, args.hfrm_ch, args.hfrm_kernel), flush=True)

    loss_weights = None
    amp_dtype = get_amp_dtype(args)
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp_dtype == 'fp16'))
    
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_eval = transforms.Compose([transforms.ToTensor()])

    train_dataset = Stage1_TrainDataset(data_path=args.trainroot, transform=transform_train, dataset=args.dataset, img_size=args.img_size, aug=args.aug)
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=data_generator,
    )
    


    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    
    param_groups = model.get_parameter_groups()
    optim_params = [
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ]

    optimizer = torchutils.PolyOptimizer(optim_params, lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    
    if args.weights[-7:] == '.params':
        weights_dict = importlib.import_module('network.resnet38d').convert_mxnet_to_torch(args.weights)
        model.load_state_dict(weights_dict, strict=False)
    elif args.weights[-4:] == '.pth':
        model.load_state_dict(torch.load(args.weights), strict=False)
        
    avg_meter = pyutils.AverageMeter('loss_cls', 'loss_adapt', 'loss_seg', 'loss_aer', 'avg_ep_EM', 'avg_ep_acc')
    timer = pyutils.Timer("Session started: ")
    best_val_miou = None
    eval_history = []
    os.makedirs(args.save_folder, exist_ok=True)

    for ep in range(args.max_epoches):
        model.train()
        ep_count = ep_EM = ep_acc = 0
        

        
        for iter, (filename, img, label) in enumerate(train_data_loader):
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out_56, out_28_1, out_28_2, out_deep, y_deep, cam_56, cam_28_1, cam_28_2, cam_deep, feat_56 = model(img)
                loss_w_56, loss_w_28_1, loss_w_28_2, loss_w_deep = get_loss_weights(args)
                loss_cls = (
                    loss_w_56 * asym_cls_loss(out_56, label, args.neg_weight)
                    + loss_w_28_1 * asym_cls_loss(out_28_1, label, args.neg_weight)
                    + loss_w_28_2 * asym_cls_loss(out_28_2, label, args.neg_weight)
                    + loss_w_deep * asym_cls_loss(out_deep, label, args.neg_weight)
                )
                loss = loss_cls
                if args.pmst_weight > 0.0 and ep >= args.pmst_start_epoch:
                    loss_seg = pmst_loss([cam_56, cam_28_1, cam_28_2, cam_deep], label,
                                         thr_lo=args.pmst_thr_lo, thr_hi=args.pmst_thr,
                                         temp=args.pmst_temp)
                    loss = loss + args.pmst_weight * loss_seg
                else:
                    loss_seg = torch.zeros((), device=img.device)
                if args.aer_weight > 0.0:
                    loss_aer = aer_loss([cam_56, cam_28_1, cam_28_2, cam_deep], label, q=args.aer_q)
                    loss = loss + args.aer_weight * loss_aer
                else:
                    loss_aer = torch.zeros((), device=img.device)
            
            loss_adapt_val = torch.zeros((), device=img.device)
            
            # Metrics
            prob = y_deep.detach().float().cpu().numpy()
            gt = label.detach().float().cpu().numpy()
            for num, one in enumerate(prob):
                ep_count += 1
                pass_cls = np.where(one > args.train_cls_thr)[0]
                true_cls = np.where(gt[num] == 1)[0]
                if np.array_equal(pass_cls, true_cls): ep_EM += 1
                ep_acc += compute_acc(pass_cls, true_cls)
            
            avg_meter.add({'loss_cls': loss_cls.item(), 'loss_adapt': loss_adapt_val.item(),
                           'loss_seg': loss_seg.item(), 'loss_aer': loss_aer.item()})
            
            optimizer.zero_grad()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            if (optimizer.global_step) % 100 == 0 and (optimizer.global_step) != 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('Epoch:%2d' % ep,
                      'Iter:%5d/%5d' % (optimizer.global_step, max_step),
                      'L_Cls:%.4f' % avg_meter.get('loss_cls'),
                      'L_Adpt:%.4f' % avg_meter.get('loss_adapt'),
                      'L_Seg:%.4f' % avg_meter.get('loss_seg'),
                      'L_AER:%.4f' % avg_meter.get('loss_aer'),
                      'Tr_EM:%.4f' % round(ep_EM/ep_count, 4),
                      'Tr_Acc:%.4f' % round(ep_acc/ep_count, 4),
                      'lr: %.4f' % optimizer.param_groups[0]['lr'],
                      'Fin:%s' % timer.str_est_finish(), flush=True)


        checkpoint_path = get_checkpoint_path(args)
        if args.save_checkpoints:
            torch.save(model.state_dict(), checkpoint_path)
        if args.save_last_k_checkpoints > 0 and (ep + 1) > args.max_epoches - args.save_last_k_checkpoints:
            epoch_checkpoint_path = os.path.join(args.save_folder, f"stage1_epoch_{ep + 1:04d}.pth")
            torch.save(model.state_dict(), epoch_checkpoint_path)
        do_eval = args.eval_every > 0 and ((ep + 1) % args.eval_every == 0)
        if do_eval:
            state_dict = None if args.save_checkpoints else model.state_dict()
            val_score = test_phase(args, dataroot=args.valroot, split_name='val', checkpoint_path=checkpoint_path, state_dict=state_dict)
            test_score = test_phase(args, dataroot=args.testroot, split_name='test', checkpoint_path=checkpoint_path, state_dict=state_dict)
            val_miou = val_score.get('Mean IoU') if val_score is not None else None
            test_miou = test_score.get('Mean IoU') if test_score is not None else None
            eval_record = {
                'epoch': ep + 1,
                'val_mean_iou': val_miou,
                'test_mean_iou': test_miou,
                'val_mean_dice': val_score.get('Mean Dice') if val_score is not None else None,
                'test_mean_dice': test_score.get('Mean Dice') if test_score is not None else None,
            }
            eval_history.append(eval_record)
            print('[Eval]', json.dumps(eval_record, sort_keys=True), flush=True)
            if val_miou is not None and (best_val_miou is None or val_miou > best_val_miou):
                best_val_miou = val_miou

    return {'best_val_miou': best_val_miou, 'eval_history': eval_history}


def test_phase(args, dataroot=None, split_name='test', checkpoint_path=None, state_dict=None):
    model = getattr(importlib.import_module(args.network), 'Net_CAM')(
        n_class=args.n_class, hfrm_gsr=args.hfrm_gsr, hfrm_ch=args.hfrm_ch,
        hfrm_kernel=args.hfrm_kernel)
    model = model.cuda()
    if dataroot is None:
        dataroot = args.testroot
    if state_dict is None:
        if checkpoint_path is None:
            checkpoint_path = get_checkpoint_path(args)
        weights_dict = torch.load(checkpoint_path)
    else:
        weights_dict = state_dict
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    score = infer(model, dataroot, args.n_class, args, thr=get_infer_thr(args), cam_weights=get_cam_weights(args))
    print(f'[{split_name}] {score}', flush=True)
    return score

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--max_epoches", default=24, type=int)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--trainroot", default='datasets/LUAD-HistoSeg/training/', type=str)
    parser.add_argument("--testroot", default='datasets/LUAD-HistoSeg/test/', type=str)
    parser.add_argument("--valroot", default='datasets/LUAD-HistoSeg/val/', type=str)

    parser.add_argument("--dataset", default='luad', type=str)
    parser.add_argument("--img_size", default=224, type=int)

    parser.add_argument("--save_folder", default='checkpoints', type=str)
    parser.add_argument("--checkpoint_name", default='stage1_last.pth', type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--eval_every", default=1, type=int)
    parser.add_argument("--evaluate_only", "--evaluate-only", action="store_true")
    parser.add_argument("--infer_thr", default=None, type=float)
    parser.add_argument("--cam_w_28_1", default=0.6, type=float)
    parser.add_argument("--cam_w_28_2", default=0.2, type=float)
    parser.add_argument("--cam_w_deep", default=0.2, type=float)
    parser.add_argument("--loss_w_56", default=0.1, type=float)
    parser.add_argument("--loss_w_28_1", default=0.15, type=float)
    parser.add_argument("--loss_w_28_2", default=0.25, type=float)
    parser.add_argument("--loss_w_deep", default=0.5, type=float)
    parser.add_argument("--train_cls_thr", default=0.2, type=float)
    parser.add_argument("--no-gsr", dest="hfrm_gsr", action="store_false", default=True,
                        help="Disable the Global Semantic Rectification (veto) branch of HFRM")
    parser.add_argument("--no-ch", dest="hfrm_ch", action="store_false", default=True,
                        help="Disable the Contextual Homogenization branch of HFRM")
    parser.add_argument("--hfrm_kernel", "--hfrm-kernel", dest="hfrm_kernel", default=15, type=int,
                        help="Large-kernel size K of the Contextual Homogenization branch")
    parser.add_argument("--pmst_weight", "--pmst-weight", dest="pmst_weight", default=0.0, type=float,
                        help="PMST 单阶段伪掩码自训练损失权重（0 表示关闭）")
    parser.add_argument("--pmst_thr", "--pmst-thr", dest="pmst_thr", default=0.7, type=float,
                        help="PMST 伪掩码高置信阈值（上界）")
    parser.add_argument("--pmst_thr_lo", "--pmst-thr-lo", dest="pmst_thr_lo", default=0.3, type=float,
                        help="PMST 伪掩码低置信阈值（下界，之间线性加权）")
    parser.add_argument("--pmst_temp", "--pmst-temp", dest="pmst_temp", default=1.0, type=float,
                        help="PMST 伪掩码 softmax 温度")
    parser.add_argument("--pmst_start_epoch", "--pmst-start-epoch", dest="pmst_start_epoch", default=5, type=int,
                        help="从第几个 epoch 开始启用 PMST（前期 CAM 不可靠）")
    parser.add_argument("--aer_weight", "--aer-weight", dest="aer_weight", default=0.0, type=float,
                        help="AER 对抗擦除正则的损失权重（0 表示关闭）")
    parser.add_argument("--aer_q", "--aer-q", dest="aer_q", default=0.3, type=float,
                        help="AER 每个尺度每类擦除的峰值像素比例 q")
    parser.add_argument("--neg_weight", "--neg-weight", dest="neg_weight", default=1.0, type=float,
                        help="非对称存在性损失中负类（不存在类）的权重；1.0=原始 multilabel_soft_margin_loss")
    parser.add_argument("--aug", action="store_true", default=False,
                        help="启用官方管线缺失的旋转/尺度/颜色增强（默认关闭，保持官方行为）")
    parser.add_argument("--amp_dtype", "--amp-dtype", default="bf16", choices=["none", "bf16", "fp16"], type=str)
    parser.add_argument("--save_checkpoints", "--save-checkpoints", dest="save_checkpoints", action="store_true", default=True)
    parser.add_argument("--no-save_checkpoints", "--no-save-checkpoints", dest="save_checkpoints", action="store_false")
    parser.add_argument("--save_last_k_checkpoints", "--save-last-k-checkpoints", default=5, type=int)
    args = parser.parse_args()
    if args.evaluate_only:
        set_seed(args.seed)
        test_phase(args, dataroot=args.testroot, split_name="test_tta", checkpoint_path=args.weights)
    else:
        os.makedirs(args.save_folder, exist_ok=True)
        start_time = time.time()
        train_phase(args)
        print(f"Total Training Time: {time.time() - start_time - time_test:.2f}s")
