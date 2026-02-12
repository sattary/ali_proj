import os, glob, argparse, random, time
import numpy as np
import matplotlib

# Handle headless plotting (no GUI)
HEADLESS = not os.environ.get("DISPLAY")
if HEADLESS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# try imports for .mat loading
try:
    import h5py
except Exception:
    h5py = None
try:
    from scipy import io as spio
except Exception:
    spio = None


# -----------------------
# Utils
# -----------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def meshgrid_ij(x, y, **kw):
    # torch.meshgrid had API change in newer PyTorch
    try:
        return torch.meshgrid(x, y, indexing='ij', **kw)
    except TypeError:
        return torch.meshgrid(x, y)

def _to_np(x):
    return x.detach().float().cpu().numpy()


# -----------------------
# Sobel / gradient helper
# -----------------------
class FixedSobel(nn.Module):
    def __init__(self):
        super().__init__()
        gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        gy = gx.t().contiguous()
        self.register_buffer("gx", gx.view(1,1,3,3))
        self.register_buffer("gy", gy.view(1,1,3,3))
    def forward(self, x):
        # x: [B,1,H,W] (or [B,C,H,W]; we repeat filters per-channel)
        gx = self.gx.to(x.device, x.dtype)
        gy = self.gy.to(x.device, x.dtype)
        C = x.shape[1]
        kx, ky = gx.repeat(C,1,1,1), gy.repeat(C,1,1,1)
        px = F.pad(x, (1,1,1,1), mode="reflect")
        return F.conv2d(px,kx,groups=C), F.conv2d(px,ky,groups=C)


# -----------------------
# Laplacian for smoothness
# -----------------------
def laplacian(u):
    # u: [B,1,H,W]
    u_pad = F.pad(u, (1,1,1,1), mode="reflect")
    return (
        -4*u_pad[...,1:-1,1:-1]
        +  u_pad[...,1:-1,2:]
        +  u_pad[...,1:-1,:-2]
        +  u_pad[...,2:,1:-1]
        +  u_pad[...,:-2,1:-1]
    )

def adaptive_curvature_loss(phi_pred, conf_mask):
    """
    Smoothness / curvature penalty, weighted by conf_mask in [0,1].
    phi_pred:  [B,1,H,W]
    conf_mask: [B,1,H,W] (here we'll pass ones, but keep it generic)
    """
    curv = laplacian(phi_pred).abs()   # [B,1,H,W]
    wcurv = conf_mask * curv
    return wcurv.mean()


# -----------------------
# Dataset
# -----------------------
def _to_chw(a: np.ndarray) -> np.ndarray:
    """
    Force [C,H,W].
    Accepts:
      [H,W] -> [1,H,W]
      [H,W,1] -> [1,H,W]
      or already [1,H,W] / [C,H,W].
    """
    a = np.array(a)
    if a.ndim == 2:
        out = a[None, ...]  # [1,H,W]
    elif a.ndim == 3:
        # Heuristic: if channel is last and small, assume [H,W,C]
        if a.shape[-1] <= 8 and a.shape[0] >= 16 and a.shape[1] >= 16:
            out = np.transpose(a, (2,0,1))  # [C,H,W]
        else:
            out = a  # hope already [C,H,W]
    else:
        raise ValueError(f"Unsupported array ndim={a.ndim}")
    return np.ascontiguousarray(out)

class MatPhaseDataset(Dataset):
    """
    Returns:
        I_input  [2,H,W]  where:
            I_input[0] = normalized interferogram (z-score per sample)
            I_input[1] = phi_hint map (broadcast scalar ref phase)
        phi_gt   [1,H,W]  ground truth absolute/unwrapped phase (radians)
        I_raw    [1,H,W]  raw interferogram (for optional intensity weighting)

    Notes:
        phi_hint is built using the GT phase at a reference pixel (e.g. center).
        We broadcast that scalar to the whole image as a prior anchor.
    """
    def __init__(self, paths, I_key="I", phi_key="dphi"):
        self.paths = paths
        self.I_key = I_key
        self.phi_key = phi_key

    def __len__(self):
        return len(self.paths)

    def _load_mat(self, path):
        # try h5py first
        if h5py is not None:
            try:
                with h5py.File(path, "r") as f:
                    I = np.array(f[self.I_key])
                    phi = np.array(f[self.phi_key])
                    return I, phi
            except Exception:
                pass
        # fallback scipy.io
        if spio is not None:
            d = spio.loadmat(path)
            return np.array(d[self.I_key]), np.array(d[self.phi_key])
        raise RuntimeError("Cannot read .mat file (need h5py or scipy.io)")

    def __getitem__(self, idx):
        p = self.paths[idx]
        I_np, phi_np = self._load_mat(p)

        I_np   = _to_chw(I_np).astype(np.float32, copy=False)    # [1,H,W]
        phi_np = _to_chw(phi_np).astype(np.float32, copy=False)  # [1,H,W]

        I_raw_t  = torch.from_numpy(I_np).float()    # [1,H,W]
        phi_gt_t = torch.from_numpy(phi_np).float()  # [1,H,W] absolute target

        # normalize interferogram per-sample
        mean = I_raw_t.mean(dim=(1,2), keepdim=True)
        std  = I_raw_t.std(dim=(1,2), keepdim=True).clamp_min(1e-6)
        I_norm_t = (I_raw_t - mean) / std            # [1,H,W]

        # build phi_hint channel from GT at one reference pixel
        # we pick the center pixel as reference
        _, H, W = phi_gt_t.shape
        cy, cx = H//2, W//2
        ref_val = phi_gt_t[0, cy, cx].item()         # scalar
        phi_hint = torch.full_like(phi_gt_t, ref_val) # [1,H,W]

        # stack I_norm and phi_hint --> [2,H,W]
        I_input = torch.cat([I_norm_t, phi_hint], dim=0)

        return I_input, phi_gt_t, I_raw_t


# -----------------------
# Metrics / visualization
# -----------------------
def compute_metrics(phi_pred, phi_gt):
    """
    Both phi_pred and phi_gt are [B,1,H,W] absolute phase (unwrapped, radians).
    we measure raw difference (no wrapping).
    """
    diff = phi_pred - phi_gt
    mae  = diff.abs().mean()
    rmse = torch.sqrt((diff**2).mean().clamp_min(1e-12))
    return {'MAE': mae.detach(), 'RMSE': rmse.detach()}

def save_epoch_visuals(I_input, phi_pred_abs_aligned, phi_gt, out_dir, epoch, max_items=8):
    """
    We'll plot:
      - I_norm (channel 0 of I_input)
      - GT φ_abs
      - Predicted φ_abs (after affine alignment)
      - Error map and its mean/std
    """
    ensure_dir(out_dir)
    B = min(I_input.shape[0], max_items)

    pred_rad = phi_pred_abs_aligned.float()
    gt_rad   = phi_gt.float()

    err_all = (pred_rad - gt_rad).flatten().abs()
    emax = torch.quantile(err_all, 0.98).item() if err_all.numel()>0 else 1.0
    vmin, vmax = -emax, emax
    cmap_div = "PuOr"

    for i in range(B):
        fig, axs = plt.subplots(2,2, figsize=(10,8))

        # channel 0 is normalized interferogram
        axs[0,0].imshow(_to_np(I_input[i,0]), cmap="gray")
        axs[0,0].set_title("I_norm[0]")
        axs[0,0].axis("off")

        # GT absolute phase
        im_gt = axs[0,1].imshow(_to_np(gt_rad[i,0]), cmap="viridis")
        axs[0,1].set_title("GT φ (rad)")
        axs[0,1].axis("off")
        cb_gt = plt.colorbar(im_gt, ax=axs[0,1], fraction=0.046)
        cb_gt.set_label("φ (rad)")

        # predicted absolute phase (aligned)
        im_pred = axs[1,0].imshow(_to_np(pred_rad[i,0]), cmap="viridis")
        axs[1,0].set_title("Pred φ_abs_aligned (rad)")
        axs[1,0].axis("off")
        cb_pred = plt.colorbar(im_pred, ax=axs[1,0], fraction=0.046)
        cb_pred.set_label("φ (rad)")

        # error map
        err_raw = pred_rad[i] - gt_rad[i]  # [1,H,W]
        bias = err_raw.mean().item()
        std  = err_raw.std(unbiased=False).item()

        im_err = axs[1,1].imshow(_to_np(err_raw[0]), cmap=cmap_div, vmin=vmin, vmax=vmax)
        axs[1,1].set_title("Δφ = Pred_aligned − GT (rad)")
        axs[1,1].axis("off")
        cb_err = plt.colorbar(im_err, ax=axs[1,1], fraction=0.046)
        cb_err.set_label("Δφ (rad)")

        # annotate bias/std
        axs[1,1].text(
            0.02,0.98,
            f"mean={bias:+.3f} rad\nstd={std:.3f} rad",
            transform=axs[1,1].transAxes,
            va="top", ha="left",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.25')
        )

        plt.tight_layout()

        fig_path = os.path.join(out_dir, f"epoch{epoch:03d}_sample{i}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)


# -----------------------
# Phase supervision loss (absolute)
# -----------------------
class MAEGradCore(nn.Module):
    """
    Core supervised loss on absolute phase:
      L_mae = |phi_abs - phi_gt|
      L_grad = |∇phi_abs - ∇phi_gt|
    Weighted by conf.
    Optionally weight gradient with sqrt(I_raw).
    """
    def __init__(self, w_mae=1.0, w_grad=0.1, intensity_weighted=False):
        super().__init__()
        self.w_mae = float(w_mae)
        self.w_grad = float(w_grad)
        self.intensity_weighted = bool(intensity_weighted)
        self.sobel = FixedSobel()

    def forward(self, phi_pred_abs, phi_gt, I_raw, conf):
        # phi_pred_abs, phi_gt: [B,1,H,W]
        # conf: [B,1,H,W] in [0,1]
        # I_raw: [B,1,H,W], optional

        abs_err = (phi_pred_abs - phi_gt).abs()  # [B,1,H,W]

        pgx, pgy = self.sobel(phi_pred_abs)
        tgx, tgy = self.sobel(phi_gt)
        grad_err = (pgx - tgx).abs() + (pgy - tgy).abs()  # [B,1,H,W]

        if self.intensity_weighted and (I_raw is not None):
            wI = I_raw.clamp_min(1e-6).sqrt()
            grad_err = grad_err * wI

        abs_term  = (conf * abs_err ).mean()
        grad_term = (conf * grad_err).mean()

        total = self.w_mae * abs_term + self.w_grad * grad_term
        return total, {"mae": abs_term.detach(), "grad": grad_term.detach()}


class PhaseSupervisionLoss(nn.Module):
    """
    Wrap-friendly term is disabled (w_wrap=0.0) because we WANT absolute phase.
    """
    def __init__(self, w_mae=1.0, w_grad=0.1, w_wrap=0.0, intensity_weighted=False):
        super().__init__()
        self.core = MAEGradCore(
            w_mae=w_mae,
            w_grad=w_grad,
            intensity_weighted=intensity_weighted
        )
        self.w_wrap = float(w_wrap)

    def forward(self, phi_pred_abs, phi_gt, I_raw, conf):
        L_core, parts = self.core(phi_pred_abs, phi_gt, I_raw, conf)

        # wrapping / periodic match (disabled by setting w_wrap=0.0)
        s_pred = torch.sin(phi_pred_abs)
        c_pred = torch.cos(phi_pred_abs)
        s_gt   = torch.sin(phi_gt)
        c_gt   = torch.cos(phi_gt)
        wrap_err = torch.sqrt((s_pred - s_gt)**2 + (c_pred - c_gt)**2 + 1e-8)
        L_wrap = (conf * wrap_err).mean()

        total = L_core + self.w_wrap * L_wrap
        parts_out = {
            "mae":  parts["mae"],
            "grad": parts["grad"],
            "wrap": L_wrap.detach()
        }
        return total, parts_out


# -----------------------
# Model: UNetRes2 + global offset head
# -----------------------
class AddCoords(nn.Module):
    """
    Adds normalized x,y coordinate channels to the input.
    We'll apply it to I_input (2 channels: I_norm+phi_hint) before encoder.
    Final input to first conv = 2 + 2 coords = 4 channels.
    """
    def forward(self, x):
        # x: [B,C,H,W]
        b,c,h,w = x.shape
        yy,xx = meshgrid_ij(
            torch.linspace(-1,1,h,device=x.device,dtype=x.dtype),
            torch.linspace(-1,1,w,device=x.device,dtype=x.dtype)
        )
        coords = torch.stack([xx,yy],0).expand(b,-1,-1,-1)  # [b,2,h,w]
        return torch.cat([x,coords],1)

class Res2_DS_Block(nn.Module):
    """
    Res2-style block with depthwise splits.
    """
    def __init__(self,in_ch,out_ch,s=4,expansion=1.0,act="relu"):
        super().__init__()
        mid = max(1,int(out_ch*expansion))
        self.s = max(2,s)

        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.act   = nn.ReLU(inplace=True) if act=="relu" else nn.SiLU(inplace=True)

        # Split mid channels into s groups
        self.sizes = []
        base=mid//self.s
        rem =mid-base*self.s
        for i in range(self.s):
            self.sizes.append(base+(1 if i<rem else 0))
        self.offsets=[sum(self.sizes[:i]) for i in range(self.s)]

        self.dw = nn.ModuleList([
            nn.Conv2d(
                self.sizes[i], self.sizes[i],
                3, padding=1, groups=self.sizes[i], bias=False
            )
            for i in range(self.s)
        ])

        self.pw  = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = (in_ch!=out_ch)
        if self.shortcut:
            self.proj=nn.Conv2d(in_ch,out_ch,1,bias=False)

    def forward(self,x):
        y = self.act(self.bn1(self.conv1(x)))
        outs=[]
        for i in range(self.s):
            c0,c1 = self.offsets[i], self.offsets[i]+self.sizes[i]
            xi = y[:,c0:c1]
            zi = self.act(self.dw[i](xi if i==0 else (xi+outs[i-1])))
            outs.append(zi)
        y2 = torch.cat(outs,1)
        y2 = self.bn2(self.pw(y2))
        if self.shortcut:
            x = self.proj(x)
        return self.act(x + y2)

class UpBlockRes2(nn.Module):
    def __init__(self,in_ch_cat,out_ch,s=4,expansion=1.0,act="relu"):
        super().__init__()
        self.conv=Res2_DS_Block(in_ch_cat,out_ch,s,expansion,act)
    def forward(self,x,skip):
        x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        return self.conv(torch.cat([x,skip],1))

class UNetRes2_AbsPhase(nn.Module):
    """
    Input:  I_input [B,2,H,W] = [I_norm, phi_hint]
            We will AddCoords -> becomes 4 channels going in.
    Output heads:
        - phi_raw map [B,1,H,W] : local phase structure
        - a_pred     [B,1,H,W] : (unused optional background term)
        - b_pred_raw [B,1,H,W] : (softplus -> amplitude >=0) [not used in loss now]
        - conf_logit [B,1,H,W] : confidence (sigmoid) [not used in loss now]
        - k_off      [B,1,1,1] : global scalar offset to add to phi_raw
    Final raw supervised prediction before alignment:
        phi_abs = phi_raw + k_off
    """
    def __init__(self, in_ch=2, base=32, act="relu", final_dropout=0.2):
        super().__init__()
        self.addcoords = AddCoords()
        C=lambda m:int(min(base*m,1024))

        # encoder
        self.enc1 = nn.Sequential(
            Res2_DS_Block(in_ch+2, C(1), 4, 1.0, act),
            Res2_DS_Block(C(1),   C(1), 4, 1.0, act),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            Res2_DS_Block(C(1),   C(2), 4, 1.0, act),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            Res2_DS_Block(C(2),   C(4), 4, 1.0, act),
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            Res2_DS_Block(C(4),   C(8), 4, 1.0, act),
        )
        self.enc5 = nn.Sequential(
            nn.MaxPool2d(2),
            Res2_DS_Block(C(8),   C(16),4, 1.0, act),
        )

        # bottleneck
        self.bott = nn.Sequential(
            Res2_DS_Block(C(16),  C(32),4, 1.0, act),
        )

        # decoder
        self.up4 = UpBlockRes2(C(32)+C(16), C(16),4,1.0,act)
        self.up3 = UpBlockRes2(C(16)+C(8),  C(8), 4,1.0,act)
        self.up2 = UpBlockRes2(C(8)+C(4),   C(4), 4,1.0,act)
        self.up1 = UpBlockRes2(C(4)+C(2),   C(2), 4,1.0,act)
        self.up0 = UpBlockRes2(C(2)+C(1),   C(1), 4,1.0,act)

        self.final_dropout = nn.Dropout2d(final_dropout)

        # pixelwise head for [phi_raw, a_pred, b_raw, conf_logit]
        self.head_pix = nn.Conv2d(C(1), 4, 1)

        # global offset head k_off:
        # take bottleneck features and squeeze to [B,1,1,1]
        self.off_conv = nn.Conv2d(C(32), C(8), 1)
        self.off_act  = nn.ReLU(inplace=True) if act=="relu" else nn.SiLU(inplace=True)
        self.off_fc   = nn.Conv2d(C(8), 1, 1)

    def forward(self, x_in):
        # x_in: [B,2,H,W]  (I_norm, phi_hint)
        x0 = self.addcoords(x_in)   # [B,4,H,W]

        e1 = self.enc1(x0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b  = self.bott(e5)

        d4 = self.up4(b,  e5)
        d3 = self.up3(d4, e4)
        d2 = self.up2(d3, e3)
        d1 = self.up1(d2, e2)
        d0 = self.up0(d1, e1)

        d0 = self.final_dropout(d0)

        pix_out = self.head_pix(d0)            # [B,4,H,W]
        phi_raw      = pix_out[:,0:1]          # [B,1,H,W]
        a_pred       = pix_out[:,1:2]          # optional / unused
        b_pred_raw   = pix_out[:,2:3]          # softplus if we ever want amplitude
        conf_logit   = pix_out[:,3:4]          # sigmoid if we ever want confidence

        # global offset from bottleneck
        z = self.off_conv(b)
        z = self.off_act(z)
        k_off = self.off_fc(z)                 # [B,1,H',W']
        # spatial average to get [B,1,1,1]
        k_off = k_off.mean(dim=(2,3), keepdim=True)

        return phi_raw, a_pred, b_pred_raw, conf_logit, k_off


# -----------------------
# EMA helper
# -----------------------
from copy import deepcopy
class EMA:
    def __init__(self, model, decay=0.999):
        self.m = deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad = False
        self.decay = decay
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for p_ema, p in zip(self.m.parameters(), model.parameters()):
            p_ema.data.mul_(d).add_(p.data, alpha=1-d)
        for b_ema, b in zip(self.m.buffers(), model.buffers()):
            b_ema.data.copy_(b.data)


# -----------------------
# Helpers
# -----------------------
def set_seed(s=1337):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def pick_device(arg):
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg

def affine_align(pred, gt):
    """
    Solve per-image affine fit: a*pred + c ~ gt  (least squares)
    pred, gt: [B,1,H,W]
    returns pred_aligned [B,1,H,W], and (a, c) each [B,1]
    """
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    gt_flat   = gt.view(B, -1)

    pred_mean = pred_flat.mean(dim=1, keepdim=True)  # [B,1]
    gt_mean   = gt_flat.mean(dim=1, keepdim=True)    # [B,1]

    pred_centered = pred_flat - pred_mean            # [B,N]
    gt_centered   = gt_flat - gt_mean                # [B,N]

    var_pred = (pred_centered**2).mean(dim=1, keepdim=True) + 1e-6  # [B,1]
    cov_pg   = (pred_centered * gt_centered).mean(dim=1, keepdim=True)  # [B,1]

    a = cov_pg / var_pred                            # [B,1]
    c = gt_mean - a * pred_mean                      # [B,1]

    a_map = a.unsqueeze(-1).unsqueeze(-1)            # [B,1,1,1]
    c_map = c.unsqueeze(-1).unsqueeze(-1)            # [B,1,1,1]

    pred_aligned = a_map * pred + c_map              # [B,1,H,W]
    return pred_aligned, a, c

@torch.no_grad()
def run_eval(model, loader, device, use_amp):
    """
    Evaluate using affine-aligned φ_abs.
    """
    if loader is None:
        return {'MAE': float('nan'), 'RMSE': float('nan')}
    sums = {'MAE':0.0, 'RMSE':0.0}
    n = 0
    for I_input, phi_gt, I_raw in loader:
        I_input = I_input.to(device)
        phi_gt  = phi_gt.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            phi_raw, a_pred, b_pred_raw, conf_logit, k_off = model(I_input)
            # raw absolute phase before alignment
            phi_abs = phi_raw + k_off  # [B,1,H,W]

        # align per-image affine
        phi_abs_aligned, a_batch, c_batch = affine_align(phi_abs, phi_gt)

        m = compute_metrics(phi_abs_aligned, phi_gt)
        bs = I_input.size(0)
        for k in sums:
            sums[k] += float(m[k]) * bs
        n += bs
    if n == 0:
        return {'MAE': float('nan'), 'RMSE': float('nan')}
    return {k: sums[k]/n for k in sums}


def smart_split(paths, seed=1337, val_frac=0.1):
    """
    Shuffle filepaths and split into train / val.
    """
    n = len(paths)
    if n <= 1:
        return paths, []
    rng = random.Random(seed)
    paths = paths[:]  # copy
    rng.shuffle(paths)
    val_count = max(1, int(round(val_frac * n)))
    train_paths = paths[:-val_count]
    val_paths   = paths[-val_count:]
    return train_paths, val_paths


# -----------------------
# Training
# -----------------------
def train(args):
    set_seed(args.seed)

    device = torch.device(pick_device(args.device))
    use_cuda = (device.type == 'cuda')
    use_amp = (use_cuda and args.use_amp)

    print(f"[startup] device={device} | amp={use_amp} | batch={args.batch_size} | workers={args.workers}")

    ensure_dir(args.out_dir)
    ensure_dir(args.vis_dir)

    data_glob = os.path.join(args.data_dir, args.pattern)
    paths = sorted(glob.glob(data_glob))
    print(f"[data] searching {data_glob}, found {len(paths)} files")
    assert paths, f"No files match {args.data_dir}/{args.pattern}"

    train_paths, val_paths = smart_split(paths, seed=args.seed, val_frac=0.1)

    train_ds = MatPhaseDataset(train_paths, I_key=args.I_key, phi_key=args.phi_key)
    val_ds   = MatPhaseDataset(val_paths,   I_key=args.I_key, phi_key=args.phi_key) if len(val_paths)>0 else None

    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_cuda,
        drop_last=True
    )
    if args.workers > 0:
        dl_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_ds, **dl_kwargs)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_cuda
    ) if val_ds is not None else None

    # model
    model = UNetRes2_AbsPhase(
        in_ch=2,
        base=args.base,
        act=args.activation,
        final_dropout=args.final_dropout
    ).to(device)

    # losses
    loss_sup = PhaseSupervisionLoss(
        w_mae=args.w_mae,
        w_grad=args.w_grad,
        w_wrap=args.w_wrap,
        intensity_weighted=args.int_wgrad
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, args.epochs),
        eta_min=args.eta_min
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMA(model, decay=args.ema_decay)

    best_mae = float('inf')

    # live plot (optional)
    if not HEADLESS:
        plt.ion()
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True)

    epochs_x = []
    train_losses = []
    val_maes = []

    try:
        for epoch in range(1, args.epochs+1):
            t0 = time.time()
            model.train()
            run_loss = 0.0
            cnt = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
            for I_input, phi_gt, I_raw in pbar:
                I_input = I_input.to(device)
                phi_gt  = phi_gt.to(device)
                I_raw   = I_raw.to(device)

                opt.zero_grad(set_to_none=True)

                try:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        phi_raw, a_pred, b_pred_raw, conf_logit, k_off = model(I_input)

                        # predicted raw absolute phase before alignment
                        phi_abs = phi_raw + k_off  # [B,1,H,W]

                        # per-image affine align
                        phi_abs_align, a_batch, c_batch = affine_align(phi_abs, phi_gt)

                        # post-process heads if we ever want them
                        # b_pred    = F.softplus(b_pred_raw)            # >=0 (unused right now)
                        # conf_pred = torch.sigmoid(conf_logit)         # [0,1] (unused right now)

                        # training confidence mask
                        conf_used = torch.ones_like(phi_abs_align)

                        # 1) absolute phase supervision (on aligned prediction)
                        L_phase, parts = loss_sup(
                            phi_abs_align,
                            phi_gt,
                            I_raw if args.int_wgrad else None,
                            conf_used
                        )

                        # 2) curvature smoothness on aligned phase
                        L_curv = args.w_curv * adaptive_curvature_loss(phi_abs_align, conf_used)

                        # final total loss
                        loss = (
                            args.w_data * L_phase +
                            L_curv
                        )

                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip,
                        error_if_nonfinite=False
                    )
                    scaler.step(opt)
                    scaler.update()
                    ema.update(model)

                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print("CUDA OOM. Try smaller --batch-size.")
                        torch.cuda.empty_cache()
                        plt.close('all')
                        return
                    else:
                        raise

                run_loss += float(loss.item()) * I_input.size(0)
                cnt += I_input.size(0)

                pbar.set_postfix({
                    "tot":  f"{float(loss.item()):.4f}",
                    "mae":  f"{float(parts['mae']):.4f}",
                    "grad": f"{float(parts['grad']):.4f}",
                })

            sched.step()
            train_loss = run_loss / max(1, cnt)

            # validation
            cur_val_mae = np.nan
            if (epoch % args.val_interval == 0) and (val_loader is not None):
                eval_stats = run_eval(
                    ema.m,
                    val_loader,
                    device,
                    use_amp
                )
                cur_val_mae = eval_stats['MAE']

                # dump visuals using EMA model on a small batch from val
                try:
                    I_input_v, phi_gt_v, I_raw_v = next(iter(val_loader))
                    I_input_v = I_input_v.to(device)
                    phi_gt_v  = phi_gt_v.to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        phi_raw_v, a_pred_v, b_pred_raw_v, conf_logit_v, k_off_v = ema.m(I_input_v)
                        phi_abs_v = phi_raw_v + k_off_v

                    # align for visualization
                    phi_abs_v_align, a_dbg, c_dbg = affine_align(phi_abs_v, phi_gt_v)

                    # optional debug prints for range *after* alignment
                    pred_min = float(phi_abs_v_align.min().item())
                    pred_max = float(phi_abs_v_align.max().item())
                    gt_min   = float(phi_gt_v.min().item())
                    gt_max   = float(phi_gt_v.max().item())
                    print(f"[val debug] after align: pred_min={pred_min:.3f} pred_max={pred_max:.3f} | gt_min={gt_min:.3f} gt_max={gt_max:.3f}")

                    save_epoch_visuals(
                        I_input_v.cpu(),
                        phi_abs_v_align.detach().cpu(),
                        phi_gt_v.cpu(),
                        args.vis_dir,
                        epoch,
                        args.vis_max
                    )
                except Exception as e:
                    print("Warning: saving visuals failed:", e)

                print(
                    f"Epoch {epoch} | train={train_loss:.4f} | "
                    f"val MAE={eval_stats['MAE']:.4f} RMSE={eval_stats['RMSE']:.4f} | "
                    f"time={time.time()-t0:.1f}s"
                )

                if eval_stats['MAE'] < best_mae:
                    best_mae = eval_stats['MAE']
                    torch.save(
                        {
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'model_ema': ema.m.state_dict()
                        },
                        os.path.join(args.out_dir, "best.pth")
                    )
            else:
                print(
                    f"Epoch {epoch} | train={train_loss:.4f} | "
                    f"time={time.time()-t0:.1f}s"
                )

            # rolling checkpoint every epoch
            torch.save(
                {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'model_ema': ema.m.state_dict()
                },
                os.path.join(args.out_dir, "final.pth")
            )

            # live plot update
            epochs_x.append(epoch)
            train_losses.append(train_loss)
            val_maes.append(cur_val_mae)

            ax.clear()
            ax.grid(True)
            ax.plot(epochs_x, train_losses, label="Train Loss")
            ax.plot(epochs_x, val_maes, label="Val MAE")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.legend()

            if not HEADLESS:
                fig.canvas.draw()
                fig.canvas.flush_events()

        print("Done. Best MAE:", best_mae)

    finally:
        if not HEADLESS:
            plt.ioff()
        fig.savefig(os.path.join(args.out_dir, "training_curve.png"), dpi=150)
        plt.close(fig)


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data-dir", default="F:/unet/U",
                   help="Folder containing .mat files with keys I and dphi")
    p.add_argument("--pattern", default="*.mat",
                   help="Glob inside data-dir, e.g. '*.mat'")
    p.add_argument("--I-key", default="I",
                   help="Key for interferogram in .mat")
    p.add_argument("--phi-key", default="dphi",
                   help="Key for ground truth unwrapped phase (radians) in .mat")

    # logging / outputs
    p.add_argument("--out-dir", default="F:/unet/runs_affine_align",
                   help="Where to write checkpoints / curve")
    p.add_argument("--vis-dir", default="F:/unet/viz_affine_align",
                   help="Where to write visualizations per val epoch")

    # model / training config
    p.add_argument("--base", type=int, default=16,
                   help="UNetRes2 base channel multiplier")
    p.add_argument("--final-dropout", type=float, default=0.3,
                   help="Dropout before pixelwise head")
    p.add_argument("--activation", choices=["relu","silu"], default="relu",
                   help="Conv block activation")

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eta-min", type=float, default=1e-6,
                   help="CosineAnnealingLR minimum LR")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0,
                   help="Global grad norm")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay for teacher model")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader workers (0 = no multiprocessing)")
    p.add_argument("--device", type=str, default="auto",
                   help="'auto', 'cuda', or 'cpu'")
    p.add_argument("--seed", type=int, default=1337)

    # precision / amp
    p.add_argument("--use-amp", action="store_true",
                   help="Enable automatic mixed precision (AMP) on CUDA")

    # supervised absolute phase terms
    p.add_argument("--w-mae", type=float, default=1.0,
                   help="Weight on |phi_abs - phi_gt| term")
    p.add_argument("--w-grad", type=float, default=0.1,
                   help="Weight on |∇phi_abs-∇phi_gt| term")
    p.add_argument("--w-wrap", type=float, default=0.0,
                   help="Periodic wrap term weight (keep 0.0 for absolute phase)")
    p.add_argument("--int-wgrad", action="store_true",
                   help="Weight grad mismatch by sqrt(raw I)")

    # smoothness / weighting
    p.add_argument("--w-curv", type=float, default=0.003,
                   help="Weight on curvature smoothness of aligned phi_abs")
    p.add_argument("--w-data", type=float, default=1.0,
                   help="Global multiplier on supervised phase loss")

    # eval / viz
    p.add_argument("--vis-max", type=int, default=8,
                   help="How many samples to dump per val epoch")
    p.add_argument("--val-interval", type=int, default=1,
                   help="Run validation/visuals every N epochs")

    return p.parse_args()


# -----------------------
# main
# -----------------------
if __name__ == "__main__":
    args = parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(args.vis_dir)

    # wipe stale outputs if you want a clean run
    for fn in ["final.pth","best.pth","training_curve.png"]:
        fp = os.path.join(args.out_dir, fn)
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except Exception:
                pass

    train(args)
