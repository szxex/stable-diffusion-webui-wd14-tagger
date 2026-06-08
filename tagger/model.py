import os
from pathlib import Path
from typing import Any, Optional, Tuple
from dataclasses import dataclass

from PIL import Image
from modules import shared
import numpy as np

# -------------------------
# デバイス選択（CPU / GPU）
# Select execution device (CPU or GPU)
# -------------------------

# デフォルトはGPU使用
use_cpu = False
if hasattr(shared.cmd_opts, 'use_cpu'):
    use_cpu = ('all' in shared.cmd_opts.use_cpu) or (
        'interrogate' in shared.cmd_opts.use_cpu)

# =========================
# Config (Immutable)
# 設定クラス（不変）
# =========================

@dataclass(frozen=True)
class ModelParam:
    # 入力画像サイズ
    # Input image size
    image_size: int = 448

    # テンソルレイアウト（NCHW / NHWC）
    # Tensor layout format
    layout: str = "NCHW"

    # カラーフォーマット（RGB / BGR）
    # Color format
    color_format: str = "RGB"

    # パディング色
    # Padding color
    pad_color: Tuple[int, int, int] = (255, 255, 255)

    # 正規化パラメータ
    # Normalization parameters
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    # 正規化を使用するか
    # Whether to use normalization
    use_normalize: bool = True

    # マスクを使用するか
    # Whether to use mask
    use_mask: bool = False


# =========================
# Preprocess Pipeline
# 画像前処理パイプライン
# =========================

class ImagePreprocessor:

    @staticmethod
    def remove_alpha(image: Image.Image) -> Image.Image:
        # アルファチャンネル除去（透明背景を白に）
        # Remove alpha channel (convert transparency to white background)
        if image.mode == "RGBA":
            bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            bg.paste(image, mask=image)
            return bg.convert("RGB")
        return image.convert("RGB")

    @staticmethod
    def resize_keep_aspect(image: Image.Image, size: int):
        # アスペクト比を維持したリサイズ
        # Resize while preserving aspect ratio
        w, h = image.size
        aspect = w / h

        if aspect > 1:
            new_w = size
            new_h = int(size / aspect)
        else:
            new_h = size
            new_w = int(size * aspect)

        return image.resize((new_w, new_h), Image.BICUBIC), (new_w, new_h)

    @staticmethod
    def pad_to_square(image: Image.Image, size: int, pad_color):
        # 正方形にパディング
        # Pad image to square
        canvas = Image.new("RGB", (size, size), pad_color)

        w, h = image.size
        x = (size - w) // 2
        y = (size - h) // 2

        canvas.paste(image, (x, y))
        return canvas, (x, y)

    @staticmethod
    def to_numpy(image: Image.Image) -> np.ndarray:
        # PIL → NumPy変換
        # Convert PIL image to NumPy array
        return np.asarray(image, dtype=np.float32)

    @staticmethod
    def convert_color(arr: np.ndarray, fmt: str):
        # RGB ↔ BGR変換
        # Convert color format (RGB ↔ BGR)
        if fmt == "BGR":
            return arr[:, :, ::-1]
        return arr

    @staticmethod
    def normalize(arr: np.ndarray, mean, std):
        # 0-255 → 0-1 → 正規化
        # Normalize pixel values
        arr /= 255.0

        if mean is not None and std is not None:
            mean = np.array(mean, dtype=np.float32)
            std = np.array(std, dtype=np.float32)
            return (arr - mean) / std
        return arr

    @staticmethod
    def transpose(arr: np.ndarray, layout: str):
        # NHWC → NCHW変換
        # Convert layout if needed
        if layout == "NCHW":
            return arr.transpose(2, 0, 1)
        return arr

    @staticmethod
    def create_mask(size: int, bbox):
        # 有効領域マスク生成
        # Create valid region mask
        mask = np.ones((size, size), dtype=bool)
        x, y, w, h = bbox
        mask[y:y + h, x:x + w] = False
        return mask

    @classmethod
    def run(cls, image: Image.Image, param: ModelParam):
        # 前処理パイプライン実行
        # Execute preprocessing pipeline

        image = cls.remove_alpha(image)

        image, (w, h) = cls.resize_keep_aspect(image, param.image_size)
        image, (x, y) = cls.pad_to_square(image, param.image_size, param.pad_color)

        arr = cls.to_numpy(image)
        arr = cls.convert_color(arr, param.color_format)

        if param.use_normalize:
            arr = cls.normalize(arr, param.normalize_mean, param.normalize_std)

        arr = cls.transpose(arr, param.layout)

        mask = None
        if param.use_mask:
            mask = cls.create_mask(param.image_size, (x, y, w, h))

        # バッチ次元追加
        # Add batch dimension
        return np.expand_dims(arr.astype(np.float32), 0), mask


# =========================
# Base Wrapper
# モデルラッパ基底クラス
# =========================

class BaseModel:

    def __init__(self, param: ModelParam):
        # モデル設定保持
        # Store model parameters
        self.param = param

    def preprocess(self, image: Image.Image):
        # 共通前処理
        # Common preprocessing
        return ImagePreprocessor.run(image, self.param)

    def __call__(self, *args, **kwargs):
        # 推論はサブクラスで実装
        # Must be implemented in subclass
        raise NotImplementedError


# =========================
# ONNX Wrapper
# ONNXモデルラッパ
# =========================

class ONNXModel(BaseModel):

    def __init__(self, model_path: str, param: Optional[ModelParam] = None):
        import onnxruntime as ort

        param = param or ModelParam()
        super().__init__(param)

        # 実行プロバイダ選択
        # Select execution providers
        if use_cpu:
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # セッション作成
        # Create inference session
        self.session = ort.InferenceSession(model_path, providers=providers)

        # 入出力名取得
        # Get input/output names
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        # モデル形状から自動設定
        # Auto-config from model shape
        self._auto_config_from_model()

    def _auto_config_from_model(self):
        # 入力形状からレイアウト・サイズ推定
        # Infer layout and size from input shape
        shape = self.session.get_inputs()[0].shape

        layout = self._infer_layout(shape)
        size = self._infer_size(shape, layout)

        self.param = ModelParam(
            image_size=size,
            layout=layout,
            color_format=self.param.color_format,
            pad_color=self.param.pad_color,
            normalize_mean=self.param.normalize_mean,
            normalize_std=self.param.normalize_std,
            use_normalize=self.param.use_normalize,
            use_mask=self.param.use_mask
        )

    @staticmethod
    def _infer_layout(shape):
        # レイアウト推定
        # Infer tensor layout
        if len(shape) != 4:
            return "NCHW"

        if shape[1] in [1, 3, 4]:
            return "NCHW"
        if shape[-1] in [1, 3, 4]:
            return "NHWC"

        return "NCHW"

    @staticmethod
    def _infer_size(shape, layout):
        # 入力サイズ推定
        # Infer image size
        h = shape[2] if layout == "NCHW" else shape[1]

        return h if isinstance(h, int) else 448

    def __call__(self, output_name=None, **inputs):
        # ONNX推論
        # Run ONNX inference
        feed = {}

        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"Missing input: {name}")

            x = inputs[name]

            # torch → numpy変換対応
            # Convert torch tensor to numpy if needed
            if hasattr(x, "cpu"):
                x = x.cpu().numpy()

            feed[name] = x

        outputs = self.session.run(self.output_names, feed)

        if output_name is None:
            return outputs[0]
        return outputs[self.output_names.index(output_name)]


# =========================
# TIMM Wrapper
# timmモデルラッパ
# =========================

class TimmModel(BaseModel):

    def __init__(self, model_path: str, config_path: str,
                 param: Optional[ModelParam] = None,
                 device: str = "cuda"):

        import timm
        from safetensors.torch import load_file

        param = param or ModelParam()
        super().__init__(param)

        # 実行デバイス設定
        # Set execution device
        self.device = "cpu" if device == "cpu" else "cuda"

        # configから前処理設定取得
        # Load preprocessing config
        self._param_from_config(config_path)

        # モデル生成
        # Create model
        model_dir = os.path.dirname(model_path)
        self.model = timm.create_model(f"local-dir:{model_dir}")

        # 重みロード
        # Load weights
        state_dict = load_file(model_path)
        self.model.load_state_dict(state_dict)

        self.model.eval().to(self.device)

    def _param_from_config(self, config_path):
        import json

        # config.jsonから前処理設定取得
        # Load preprocessing parameters from config
        with open(config_path, "r") as f:
            cfg = json.load(f)

        pcfg = cfg.get("pretrained_cfg", {})

        self.param = ModelParam(
            image_size=pcfg.get("input_size", [3, 448, 448])[1],
            layout="NCHW",
            color_format="BGR",
            pad_color=tuple(pcfg.get("pad_color", [255, 255, 255])),
            normalize_mean=tuple(pcfg.get("mean", [0.5, 0.5, 0.5])),
            normalize_std=tuple(pcfg.get("std", [0.5, 0.5, 0.5])),
            use_normalize=True,
            use_mask=False
        )

    def __call__(self, **inputs):
        import torch

        # 入力チェック
        # Validate input
        if "image" not in inputs:
            raise ValueError("Missing input: image")

        x = inputs["image"]

        # Tensor化
        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # バッチ次元追加
        # Add batch dimension
        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        # 推論（勾配なし）
        # Inference without gradients
        with torch.no_grad():
            out = self.model(x)

        return out.cpu().numpy()


# =========================
# Factory
# モデル生成ファクトリ
# =========================

class ModelFactory:

    @staticmethod
    def load(model_path: str, **kwargs):
        # 拡張子でモデル種別判定
        # Detect model type by file extension
        ext = Path(model_path).suffix.lower()

        if ext == ".onnx":
            return ONNXModel(model_path, **kwargs)

        if ext == ".safetensors":
            if "config_path" not in kwargs:
                raise ValueError("config_path is required for safetensors")
            return TimmModel(model_path, **kwargs)

        # 未対応フォーマット
        # Unsupported format
        raise ValueError(f"Unsupported model format: {ext}")