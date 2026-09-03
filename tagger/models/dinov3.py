from functools import lru_cache
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3Tagger(nn.Module):
    """
    DINOv3 ViT-H/16+ Tagger

    Responsibilities
    ----------------
    * DINOv3 Backbone
    * Classification Head
    * checkpoint loading
    * checkpoint format conversion

    DINOv3Model(BaseModel) はこのクラスを呼ぶだけ。
    """

    # ======================================================
    # Configuration
    # ======================================================

    PATCH_SIZE = 16

    ROPE_THETA = 100.0
    ROPE_RESCALE = 2.0

    D_MODEL = 1280
    NUM_HEADS = 20
    HEAD_DIM = D_MODEL // NUM_HEADS

    MLP_DIM = 5120
    N_LAYERS = 32
    N_REGISTERS = 4

    FEATURE_DIM = D_MODEL * (1 + N_REGISTERS)

    LAYERSCALE = 1e-6
    LN_EPS = 1e-6

    # ======================================================
    # Embedding
    # ======================================================

    class Embeddings(nn.Module):

        def __init__(self):
            super().__init__()

            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, DINOv3Tagger.D_MODEL)
            )

            self.mask_token = nn.Parameter(
                torch.zeros(1, 1, DINOv3Tagger.D_MODEL)
            )

            self.register_tokens = nn.Parameter(
                torch.zeros(
                    1,
                    DINOv3Tagger.N_REGISTERS,
                    DINOv3Tagger.D_MODEL,
                )
            )

            self.patch_embeddings = nn.Conv2d(
                3,
                DINOv3Tagger.D_MODEL,
                kernel_size=DINOv3Tagger.PATCH_SIZE,
                stride=DINOv3Tagger.PATCH_SIZE,
            )

        def forward(self, pixel_values):
            B = pixel_values.shape[0]

            dtype = self.patch_embeddings.weight.dtype

            patches = self.patch_embeddings(pixel_values.to(dtype))
            patches = patches.flatten(2).transpose(1, 2)

            cls = self.cls_token.expand(B, -1, -1)
            regs = self.register_tokens.expand(B, -1, -1)

            return torch.cat([cls, regs, patches], dim=1)

    # ======================================================
    # Attention
    # ======================================================

    class Attention(nn.Module):
        """DINOv3 Attention (HF/Meta checkpoint compatible)."""

        def __init__(self):
            super().__init__()

            self.q_proj = nn.Linear(DINOv3Tagger.D_MODEL,
                                    DINOv3Tagger.D_MODEL,
                                    bias=True)

            self.k_proj = nn.Linear(DINOv3Tagger.D_MODEL,
                                    DINOv3Tagger.D_MODEL,
                                    bias=False)

            self.v_proj = nn.Linear(DINOv3Tagger.D_MODEL,
                                    DINOv3Tagger.D_MODEL,
                                    bias=True)

            self.o_proj = nn.Linear(DINOv3Tagger.D_MODEL,
                                    DINOv3Tagger.D_MODEL,
                                    bias=True)

        @staticmethod
        def rotate_half(x):
            h = x.shape[-1] // 2
            return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


        @classmethod
        def apply_rope(cls, q, k, cos, sin):
            n_pre = 1 + DINOv3Tagger.N_REGISTERS

            q_pre = q[..., :n_pre, :]
            q_patch = q[..., n_pre:, :]

            k_pre = k[..., :n_pre, :]
            k_patch = k[..., n_pre:, :]

            q_patch = q_patch * cos + cls.rotate_half(q_patch) * sin
            k_patch = k_patch * cos + cls.rotate_half(k_patch) * sin

            q = torch.cat([q_pre, q_patch], dim=-2)
            k = torch.cat([k_pre, k_patch], dim=-2)

            return q, k

        def forward(self, x, cos, sin):
            B, N, _ = x.shape

            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            q = q.view(B, N, DINOv3Tagger.NUM_HEADS,
                    DINOv3Tagger.HEAD_DIM).transpose(1, 2)

            k = k.view(B, N, DINOv3Tagger.NUM_HEADS,
                    DINOv3Tagger.HEAD_DIM).transpose(1, 2)

            v = v.view(B, N, DINOv3Tagger.NUM_HEADS,
                    DINOv3Tagger.HEAD_DIM).transpose(1, 2)

            q, k = self.apply_rope(q, k, cos, sin)

            x = F.scaled_dot_product_attention(q, k, v)

            x = x.transpose(1, 2).reshape(B, N, DINOv3Tagger.D_MODEL)

            return self.o_proj(x)

    # ======================================================
    # SwiGLU MLP
    # ======================================================

    class GatedMLP(nn.Module):
        """SwiGLU MLP compatible with DINOv3 checkpoint."""

        def __init__(self):
            super().__init__()

            self.gate_proj = nn.Linear(
                DINOv3Tagger.D_MODEL,
                DINOv3Tagger.MLP_DIM,
                bias=True,
            )

            self.up_proj = nn.Linear(
                DINOv3Tagger.D_MODEL,
                DINOv3Tagger.MLP_DIM,
                bias=True,
            )

            self.down_proj = nn.Linear(
                DINOv3Tagger.MLP_DIM,
                DINOv3Tagger.D_MODEL,
                bias=True,
            )

        def forward(self, x):
            x = F.silu(self.gate_proj(x)) * self.up_proj(x)
            return self.down_proj(x)

    # ======================================================
    # Transformer Block
    # ======================================================

    class Block(nn.Module):

        def __init__(self):
            super().__init__()

            self.norm1 = nn.LayerNorm(
                DINOv3Tagger.D_MODEL,
                eps=DINOv3Tagger.LN_EPS,
            )

            self.attention = DINOv3Tagger.Attention()

            self.layer_scale1 = nn.Parameter(
                torch.full(
                    (DINOv3Tagger.D_MODEL,),
                    DINOv3Tagger.LAYERSCALE,
                )
            )

            self.norm2 = nn.LayerNorm(
                DINOv3Tagger.D_MODEL,
                eps=DINOv3Tagger.LN_EPS,
            )

            self.mlp = DINOv3Tagger.GatedMLP()

            self.layer_scale2 = nn.Parameter(
                torch.full(
                    (DINOv3Tagger.D_MODEL,),
                    DINOv3Tagger.LAYERSCALE,
                )
            )

        def forward(self, x, cos, sin):
            x = x + self.attention(self.norm1(x), cos, sin) * self.layer_scale1
            x = x + self.mlp(self.norm2(x)) * self.layer_scale2
            return x

    # ======================================================
    # Backbone
    # ======================================================

    class ViTH(nn.Module):

        def __init__(self):
            super().__init__()

            self.embeddings = DINOv3Tagger.Embeddings()

            self.layer = nn.ModuleList(
                [DINOv3Tagger.Block() for _ in range(DINOv3Tagger.N_LAYERS)]
            )

            self.norm = nn.LayerNorm(
                DINOv3Tagger.D_MODEL,
                eps=DINOv3Tagger.LN_EPS,
            )

        @staticmethod
        @lru_cache(maxsize=32)
        def _patch_coords_cached(h: int, w: int, device_str: str):
            device = torch.device(device_str)

            cy = torch.arange(0.5, h, dtype=torch.float32, device=device) / h
            cx = torch.arange(0.5, w, dtype=torch.float32, device=device) / w

            coords = torch.stack(
                torch.meshgrid(cy, cx, indexing="ij"),
                dim=-1,
            ).flatten(0, 1)

            coords = 2.0 * coords - 1.0
            coords = coords * DINOv3Tagger.ROPE_RESCALE

            return coords


        @classmethod
        def build_rope(cls, h_patch, w_patch, dtype, device):
            coords = cls._patch_coords_cached(h_patch, w_patch, str(device))

            inv_freq = 1.0 / (
                DINOv3Tagger.ROPE_THETA ** torch.arange(
                    0,
                    1,
                    4 / DINOv3Tagger.HEAD_DIM,
                    dtype=torch.float32,
                    device=device,
                )
            )

            angles = (
                2 * math.pi
                * coords[:, :, None]
                * inv_freq[None, None, :]
            )

            angles = angles.flatten(1, 2).tile(2)

            cos = torch.cos(angles).to(dtype).unsqueeze(0).unsqueeze(0)
            sin = torch.sin(angles).to(dtype).unsqueeze(0).unsqueeze(0)

            return cos, sin

        def forward(self, pixel_values):
            _, _, H, W = pixel_values.shape

            x = self.embeddings(pixel_values)

            cos, sin = self.build_rope(
                H // DINOv3Tagger.PATCH_SIZE,
                W // DINOv3Tagger.PATCH_SIZE,
                x.dtype,
                pixel_values.device,
            )

            for block in self.layer:
                x = block(x, cos, sin)

            return self.norm(x)

    # ======================================================
    # Head
    # ======================================================

    class LowRankHead(nn.Module):

        def __init__(self, in_dim, rank, num_tags, down_bias=False, up_bias=True):
            super().__init__()

            self.proj_down = nn.Linear(in_dim, rank, bias=down_bias)
            self.proj_up = nn.Linear(rank, num_tags, bias=up_bias)

        def forward(self, x):
            return self.proj_up(self.proj_down(x))

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(self):
        super().__init__()

        self.backbone = DINOv3Tagger.ViTH()
        self.head = None

    # ======================================================
    # Checkpoint Loader
    # ======================================================

    def load_checkpoint(self, state_dict):
        """Load DINOv3 checkpoint."""

        backbone_sd, head_sd = self._split_and_clean_state_dict(state_dict)

        num_tags = self._infer_num_tags(head_sd)

        head_module, head_state = self._build_head_from_checkpoint(
            head_sd,
            self.FEATURE_DIM,
            num_tags,
        )

        self.head = head_module

        self.backbone.load_state_dict(backbone_sd, strict=True)
        self.head.load_state_dict(head_state, strict=True)

        return num_tags

    # ======================================================
    # Private Utilities
    # ======================================================

    def _split_and_clean_state_dict(self, state_dict):
        backbone_sd = {}
        head_sd = {}

        for key, value in state_dict.items():

            if key.startswith("backbone."):
                key = key.removeprefix("backbone.")

                if key.startswith("model.layer."):
                    key = key.removeprefix("model.")

                backbone_sd[key] = value
            else:
                head_sd[key] = value

        # layer_scale.lambda1 → layer_scale
        for key in list(backbone_sd.keys()):
            if key.endswith(".lambda1"):
                backbone_sd[key[:-8]] = backbone_sd.pop(key)

        # remove rope cache
        for key in list(backbone_sd.keys()):
            if "rope_embeddings" in key:
                backbone_sd.pop(key)

        return backbone_sd, head_sd

    def _infer_num_tags(self, head_sd):
        # Dense Head
        for value in head_sd.values():
            if value.ndim == 2 and value.shape[1] == self.FEATURE_DIM:
                return value.shape[0]

        # LowRank Head
        if "proj_up.weight" in head_sd:
            return head_sd["proj_up.weight"].shape[0]

        raise RuntimeError("Cannot infer num_tags.")

    def _build_head_from_checkpoint(self, head_sd, in_dim, num_tags):

        # Dense Head
        for key, weight in head_sd.items():
            if (
                key.endswith(".weight")
                and tuple(weight.shape) == (num_tags, in_dim)
            ):
                prefix = key[:-7]
                bias_key = prefix + ".bias"

                has_bias = bias_key in head_sd

                head = nn.Linear(in_dim, num_tags,bias=has_bias)

                state = {"weight": weight}

                if has_bias:
                    state["bias"] = head_sd[bias_key]

                return head, state

        # LowRank Head
        down = head_sd["proj_down.weight"]
        up = head_sd["proj_up.weight"]

        rank = down.shape[0]

        head = DINOv3Tagger.LowRankHead(
            in_dim,
            rank,
            num_tags,
            down_bias="proj_down.bias" in head_sd,
            up_bias="proj_up.bias" in head_sd,
        )

        state = {
            "proj_down.weight": down,
            "proj_up.weight": up,
        }

        if "proj_down.bias" in head_sd:
            state["proj_down.bias"] = head_sd["proj_down.bias"]

        if "proj_up.bias" in head_sd:
            state["proj_up.bias"] = head_sd["proj_up.bias"]

        return head, state

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, pixel_values):
        hidden = self.backbone(pixel_values)

        cls = hidden[:, 0]
        regs = hidden[:, 1 : 1 + self.N_REGISTERS].flatten(1)

        features = torch.cat([cls, regs], dim=-1).float()

        if self.head is None:
            raise RuntimeError("DINOv3 head is not initialized.")

        return self.head(features)

    def to_device(self, device: torch.device):
        """
        DINOv3 の推論用 device / dtype を設定する。

        Backbone : CUDAならBF16、それ以外はFP32
        Head     : 常にFP32
        """

        if (
            device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        ):
            backbone_dtype = torch.bfloat16
        else:
            backbone_dtype = torch.float32

        self.backbone = self.backbone.to(
            device=device,
            dtype=backbone_dtype,
        )

        self.head = self.head.to(
            device=device,
            dtype=torch.float32,
        )

        self.eval()

        return backbone_dtype