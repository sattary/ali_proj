import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinUNet(nn.Module):
    """
    Swin-Transformer based U-Net for Phase Unwrapping.
    Replaces the bottleneck and key encoder/decoder stages with Swin Blocks.
    """

    def __init__(
        self,
        img_size=128,
        patch_size=4,
        in_chans=2,
        num_classes=1,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encoder Layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList(
                [
                    SwinTransformerBlock(
                        dim=int(embed_dim * 2**i_layer),
                        input_resolution=(
                            patches_resolution[0] // (2**i_layer),
                            patches_resolution[1] // (2**i_layer),
                        ),
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        shift_size=0 if (i_layer % 2 == 0) else window_size // 2,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]) + j],
                        norm_layer=norm_layer,
                    )
                    for j in range(depths[i_layer])
                ]
            )
            self.layers.append(layer)

        # Downsample layers (simple patch merging simulation for now or Conv)
        # For simplicity in this UNet, we use Strided Conv for downsampling between stages
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            self.downsamples.append(
                nn.Conv2d(
                    int(embed_dim * 2**i_layer),
                    int(embed_dim * 2 ** (i_layer + 1)),
                    kernel_size=2,
                    stride=2,
                )
            )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                int(embed_dim * 2 ** (self.num_layers - 1)),
                int(embed_dim * 2 ** (self.num_layers)),
                1,
            ),
            nn.GELU(),
        )

        # Decoder Layers (Swin blocks + Upsample)
        self.upsamples = nn.ModuleList()
        self.dec_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1, -1, -1):
            # Upsample
            if i_layer < self.num_layers - 1:
                self.upsamples.append(
                    nn.ConvTranspose2d(
                        int(embed_dim * 2 ** (i_layer + 1)),
                        int(embed_dim * 2**i_layer),
                        kernel_size=2,
                        stride=2,
                    )
                )

            # Decoder Swin Block (simplified for brevity, often symmetric to encoder)
            # Here we use simple Conv ResBlocks for decoder to save compute, making it a "Hybrid"
            self.dec_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        int(embed_dim * 2**i_layer) * 2
                        if i_layer < self.num_layers - 1
                        else int(embed_dim * 2**i_layer),
                        int(embed_dim * 2**i_layer),
                        3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(int(embed_dim * 2**i_layer)),
                    nn.ReLU(inplace=True),
                )
            )

        # Output heads
        # 1: phi_raw, 2: residue_logit
        self.head = nn.Conv2d(embed_dim, 2, 1)
        self.global_offset_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(embed_dim * 2 ** (self.num_layers)), 1),
        )

    def forward(self, x):
        # x: [B, 2, H, W]
        x = self.patch_embed(x)  # [B, L, C]
        B, L, C = x.shape
        H, W = self.patches_resolution
        x = x.transpose(1, 2).view(B, C, H, W)

        # Encoder
        skips = []
        for i in range(self.num_layers):
            # Swin Blocks
            B_, C_, H_, W_ = x.shape
            x_flat = x.flatten(2).transpose(1, 2)  # [B, L, C]
            for blk in self.layers[i]:
                x_flat = blk(x_flat)
            x = x_flat.transpose(1, 2).view(B_, C_, H_, W_)

            skips.append(x)
            if i < self.num_layers - 1:
                x = self.downsamples[i](x)

        # Bottleneck
        # x is now at lowest resolution
        b = self.bottleneck(x)  # [B, C_bot, H_bot, W_bot]

        # Global offset (from bottleneck features)
        k_off = self.global_offset_head(b).view(B, 1, 1, 1)

        # Decoder
        x = skips[-1]  # Start with deepest features
        for i in range(len(self.upsamples)):
            # Upsample
            x = self.upsamples[i](x)
            # Skip connection
            skip = skips[-(i + 2)]
            x = torch.cat([x, skip], dim=1)
            # Process
            x = self.dec_layers[i](x)

        # Final head
        out = self.head(x)  # [B, 2, H, W]
        phi_raw = out[:, 0:1]
        res_logit = out[:, 1:2]

        # Upsample to original resolution if patch_size > 1
        if self.patch_embed.patch_size[0] > 1:
            phi_raw = F.interpolate(
                phi_raw, scale_factor=self.patch_embed.patch_size[0], mode="bilinear"
            )
            res_logit = F.interpolate(
                res_logit, scale_factor=self.patch_embed.patch_size[0], mode="bilinear"
            )

        # Result tuple as expected by train loop
        # phi_raw, a_pred, b_pred_raw, conf_logit, k_off
        # We REPURPOSE conf_logit for res_logit to avoid breaking train loop signature if possible,
        # or we update train loop. Let's update train loop to be more flexible.
        return phi_raw, None, None, res_logit, k_off
