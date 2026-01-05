import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class PatchEmbed(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.proj = nn.Conv2d(config.in_chans, config.embed_dim,
                              kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, attn_drop: float, proj_drop: float):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SMIFModule(nn.Module):
    def __init__(self, channels: int, window_size: int = 5, alpha: float = 0.5, threshold: float = 0.05):
        super().__init__()
        self.window_size = window_size
        self.half = window_size // 2
        self.threshold = threshold
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.conv_fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        motion_accum = torch.zeros_like(video)
        for offset in range(1, self.half + 1):
            prev_frames = torch.roll(video, shifts=offset, dims=1)
            next_frames = torch.roll(video, shifts=-offset, dims=1)
            # Fix boundary artifacts roughly by zeroing (simple approach) or copying
            # Here keeping simple as per original code logic usually implies
            prev_frames[:, :offset] = video[:, :offset]
            next_frames[:, -offset:] = video[:, -offset:]
            
            diff_f = next_frames - video
            diff_b = video - prev_frames
            motion_accum = motion_accum + diff_f.abs() + diff_b.abs()

        motion_map = motion_accum / max(self.half, 1)
        mask = (motion_map > self.threshold).float()
        motion_map = motion_map * mask

        base = video.reshape(B * T, C, H, W)
        motion_flat = motion_map.reshape(B * T, C, H, W)
        fused = torch.cat([base, motion_flat], dim=1)
        fused = self.conv_fuse(fused)
        out = base + self.alpha.tanh() * fused
        out = out.clamp(min=-1.0, max=1.0)
        return out.view(B, T, C, H, W)

class LMIModule(nn.Module):
    def __init__(self, dim: int, reduction: int = 4, delta: float = 0.1):
        super().__init__()
        reduced_dim = max(1, dim // reduction)
        self.reduce = nn.Linear(dim, reduced_dim)
        self.expand = nn.Linear(reduced_dim, dim)
        self.temporal_mlp = nn.Sequential(
            nn.LayerNorm(reduced_dim),
            nn.Linear(reduced_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, reduced_dim),
        )
        self.delta = nn.Parameter(torch.tensor(delta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        reduced = self.reduce(x)
        
        if T > 1:
            diff_f = reduced[:, 1:] - reduced[:, :-1]
            diff_f = torch.cat([diff_f, diff_f[:, -1:]], dim=1)
            diff_b = reduced[:, :-1] - reduced[:, 1:]
            diff_b = torch.cat([diff_b[:, :1], diff_b], dim=1)
        else:
            diff_f = torch.zeros_like(reduced)
            diff_b = torch.zeros_like(reduced)

        motion = (diff_f.abs() + diff_b.abs()).mean(dim=2)
        motion = self.temporal_mlp(motion)
        
        attn = torch.sigmoid(motion).unsqueeze(2)
        attn = self.expand(attn)
        attn = attn.expand(-1, -1, N, -1)
        enhanced = x * attn
        return x + self.delta.tanh() * enhanced

class LSViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop_rate: float, 
                 attn_drop: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, True, attn_drop, drop_rate)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop_rate)
        self.drop_path2 = DropPath(drop_path)
        self.lmim = LMIModule(dim)

    def forward(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        BT, Np1, C = x.shape
        x = x.view(B, T, Np1, C)
        x = self.lmim(x)
        x = x.view(B * T, Np1, C)
        return x

class LSViTBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(config.drop_rate)
        dpr = torch.linspace(0, config.drop_path_rate, steps=config.depth).tolist()
        
        self.blocks = nn.ModuleList([
            LSViTBlock(
                dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                drop_rate=config.drop_rate, attn_drop=config.attn_drop_rate, drop_path=dpr[i]
            ) for i in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _interpolate_pos_encoding(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        num_patches = N - 1
        if num_patches == self.patch_embed.num_patches:
            return self.pos_embed
        
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = patch_pos.shape[-1]
        gs_old = int(math.sqrt(patch_pos.shape[1]))
        gs_new = int(math.sqrt(num_patches))
        
        patch_pos = patch_pos.reshape(1, gs_old, gs_old, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(gs_new, gs_new), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        x = video.reshape(B * T, C, H, W)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self._interpolate_pos_encoding(x)
        x = x + pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x, B, T)
        x = self.norm(x)
        x = x.view(B, T, x.shape[1], x.shape[2])
        return x

class LSViTForAction(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.smif = SMIFModule(config.in_chans, window_size=config.smif_window)
        self.backbone = LSViTBackbone(config)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = self.smif(video)
        feats = self.backbone(x)
        cls_tokens = feats[:, :, 0]
        pooled = cls_tokens.mean(dim=1)
        logits = self.head(pooled)
        return logits