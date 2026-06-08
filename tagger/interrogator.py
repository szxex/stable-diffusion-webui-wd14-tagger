from dataclasses import dataclass
import numpy as np
import json
from typing import Tuple, List, Dict,Callable, Set
try:
    from typing import override
except ImportError:
    from typing_extensions import override

from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download

from .model import ModelFactory, ModelParam
from .tag_loader import TagLoader

@dataclass
class TagData:
    name: str
    category: str
    score: float

# =========================
# Base Interrogator Class
# 共通ベースクラス
# =========================
class Interrogator:
    name: str
    tags: Dict[int, Dict[str, str]]
    model_categories: Set[str]
    use_sigmoid: bool

    def __init__(self, name: str, use_sigmoid: bool = False) -> None:
        # クラス名（モデル識別用）
        # Name of the interrogator (model identifier)
        self.name = name

        # タグ情報（後でロードされる）
        # Tag metadata (loaded later)
        self.tags = None

        # 使用するカテゴリ（デフォルトは general）
        # Supported categories (default: general)
        self.model_categories = ('general',)

        # 出力にシグモイドを適用するか
        # Whether to apply sigmoid to output
        self.use_sigmoid = use_sigmoid

    # -------------------------
    # HuggingFaceからファイルDL
    # Download files from HuggingFace
    # -------------------------
    def _download(self, **files) -> List[Path] | Path:
        # 指定されたファイルをHuggingFace Hubからダウンロード
        # Download specified files from HuggingFace Hub
        paths = [
            Path(hf_hub_download(**self.kwargs, filename=f))
            for f in files.values()
        ]
        return paths[0] if len(paths) == 1 else paths

    @staticmethod
    def _make_category_normalizer(categories_map) -> Callable[[str, str], str]:
        # 名前をカテゴリに変換する関数を生成
        # Create a function to normalize names to category 
        def _normalize(name: str, cat: str) -> str:
            return categories_map.get(cat, cat)
        return _normalize

    # -------------------------
    # シグモイド関数
    # sigmoid activation
    # -------------------------
    @staticmethod
    def _sigmoid(x):
        # 数値安定性のためにclipしてからsigmoid適用
        # Apply sigmoid with clipping for numerical stability
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    # 出力構築
    # Build structured output (rating / tags / characters)
    def _build_output(
            self,
            probs,
            categories=None,
            category_thresholds=None
        ) -> List[TagData]:

        items: List[TagData] = []

        # 各タグの確率を評価
        # Evaluate probability for each tag
        for idx, prob in enumerate(probs):
            prob = float(prob)

            # タグ情報取得
            # Get tag metadata
            tag_info = self.tags.get(idx)
            if tag_info is None:
                continue

            name = tag_info["name"]
            cat  = tag_info["category"]

            # カテゴリフィルタ
            # Category filtering
            if categories is not None and cat not in categories:
                continue

            # カテゴリごとの閾値設定
            # Threshold per category
            if category_thresholds and cat in category_thresholds:
                th = category_thresholds[cat]
            else:
                th = 0

            # 閾値未満は除外
            # Skip if below threshold
            if prob < th or prob < 0.01:
                continue

            # TagDataとして追加
            # Append as TagData
            items.append(TagData(category=cat, name=name, score=prob))

        return items

    # タグ情報読み込み
    # Load tag metadata
    def load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        if self.tags is None:
            self.model_categories, self.tags = self._load_tags()
        return self.model_categories, self.tags

    def _load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        # 各モデルで実装されるべきメソッド
        # Must be implemented in subclasses
        raise NotImplementedError

    # ModelとTag情報読み込み
    # Load model and tag metadata
    def load(self) -> None:
        # 既にロード済みなら何もしない
        # Do nothing if already loaded
        if hasattr(self, 'model') and self.model is not None:
            return

        # モデル読み込み
        # Load model
        self.model = self._load_model()

        # タグ読み込み
        # Load tags
        self.load_tags()

    def _load_model(self):
        # 各モデルごとの実装が必要
        # Must be implemented per model
        raise NotImplementedError()

    def unload(self) -> bool:
        # モデルをメモリから解放
        # Unload model from memory
        unloaded = False

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            unloaded = True
            print(f'Unloaded {self.name}')

        if hasattr(self, 'tags'):
            self.tags = None

        return unloaded

    # 推論処理
    # Inference process
    def interrogate(
        self,
        image: Image,
        **kwargs
    ) -> List[TagData]:

        # モデル未ロードならロード
        # Load model if not loaded
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        # オプション取得
        # Get optional parameters
        categories = kwargs.get("categories")
        category_thresholds = kwargs.get("category_thresholds")

        # 画像前処理
        # Image preprocessing
        processed_image, mask = self.model.preprocess(image)

        # コア推論処理
        # Core inference
        output = self._interrogate_core(processed_image, mask, **kwargs)

        # 出力変換
        # Convert output probabilities
        if self.use_sigmoid:
            probs = Interrogator._sigmoid(output).flatten()
        else:
            probs = output[0]

        return self._build_output(probs, categories, category_thresholds)

    # 個別コア推論処理
    # Core inference (to be implemented per model)
    def _interrogate_core(
            self,
            processed_image: np.ndarray,
            mask: np.ndarray,
            **kwargs
        ) -> np.ndarray:
        raise NotImplementedError()

# =========================
# WaifuDiffusion Interrogator
# WaifuDiffusion用タグ推論クラス
# =========================
class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
        self,
        name: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        categories_map={"0": "general", "4": "character", "9": "rating"},
        **kwargs
    ) -> None:
        # 親クラス初期化
        # Initialize base class
        super().__init__(name)

        # モデルとタグファイルのパス
        # Paths for model and tag definitions
        self.model_path = model_path
        self.tags_path = tags_path

        # カテゴリID → 名前変換
        # Convert category IDs to readable names
        self.normalizer = Interrogator._make_category_normalizer(categories_map)

        # HuggingFaceダウンロード用引数
        # Arguments for HuggingFace download
        self.kwargs = kwargs

    @override
    def _load_tags(self):
        # タグCSVをダウンロードして読み込み
        # Download and load tag CSV
        tags_path = self._download(tags=self.tags_path)
        return TagLoader.load_csv(tags_path, self.normalizer)

    @override
    def _load_model(self):
        # ONNXモデルをダウンロードしてロード
        # Download and load ONNX model
        model_path = self._download(model=self.model_path)

        # WaifuDiffusionはBGR＆非正規化入力前提
        # Model expects BGR input without normalization
        return ModelFactory.load(
            model_path,
            param=ModelParam(
                color_format="BGR",
                pad_color=(255, 255, 255),
                use_normalize=False
            )
        )

    @override
    def _interrogate_core(self, processed_image, mask, **kwargs):
        # 推論実行
        # Run inference
        return self.model(**{self.model.input_names[0]: processed_image})

# =========================
# ML-Danbooru Interrogator
# Danbooruタグ推論モデル
# =========================
class MLDanbooruInterrogator(Interrogator):

    def __init__(self, name: str,
                 model_path='ml_caformer_m36_dec-5-97527.onnx',
                 tags_path='tags.csv',
                 **kwargs):

        # sigmoidを使用（multi-label classification）
        # Use sigmoid for multi-label output
        super().__init__(name, use_sigmoid=True)

        self.model_path = model_path
        self.tags_path = tags_path

        # Danbooruはカテゴリ変換なし
        # No category normalization
        self.normalizer = None

        self.kwargs = kwargs

    @override
    def _load_tags(self):
        # タグCSV読み込み
        # Load tag CSV
        tags_path = self._download(tags=self.tags_path)
        return TagLoader.load_csv(tags_path, self.normalizer, tag_name="tag")

    @override
    def _load_model(self):
        # モデルロード（正規化あり）
        # Load model with normalization
        model_path = self._download(model=self.model_path)

        return ModelFactory.load(
            model_path,
            param=ModelParam(
                color_format="BGR",
                use_normalize=True,
                normalize_mean=None,
                normalize_std=None
            )
        )

    @override
    def _interrogate_core(self, processed_image, mask, **kwargs):
        # 推論
        # Run inference
        return self.model(**{self.model.input_names[0]: processed_image})

class OracleInterrogator(Interrogator):
    def __init__(
        self,
        name,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        preproc_path='preprocessing.json',
        **kwargs):
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.preproc_path = preproc_path
        self.normalizer = OracleInterrogator._normalize_category
        self.kwargs = kwargs

    @staticmethod
    def _normalize_category(name: str, cat: str) -> str:
        if name.lower().startswith("rating:"):
            return "rating"
        if cat == "0":
            return "general"
        return cat

    @override
    def _load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        tags_path = self._download(
            tags=self.tags_path
        )
        # CSV読み込み & 正規化
        return TagLoader.load_csv(tags_path,self.normalizer)


    @override
    def _load_model(self):
        model_path, preproc_path = self._download(
             model=self.model_path,
             preproc=self.preproc_path
        )

        with open(preproc_path, "r") as f:
            preproc = json.load(f)

        # ONNXロード
        return ModelFactory.load(
            model_path,
            param=ModelParam(
                color_format="RGB",
                pad_color=tuple(preproc["pad_color_rgb"]),
                use_normalize=True,
                normalize_mean=preproc["normalize_mean"],
                normalize_std=preproc["normalize_std"],
                use_mask=True
            )
        )

    @override
    def _interrogate_core(
        self,
        processed_image: np.ndarray,
        mask: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        # ONNX推論
        return self.model(
            **{
                self.model.input_names[0]: processed_image,
                self.model.input_names[1]: mask[None, ...]
            }
        )

# =========================
# PixAI Interrogator
# =========================
class PixAIInterrogator(Interrogator):
    def __init__(
    self,
    name,
    model_path='model.onnx',
    tags_path='selected_tags.csv',
    preproc_path='preprocess.json',
    categories_map={"0":"general","4":"character"},
    **kwargs):
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.preproc_path = preproc_path
        self.normalizer = Interrogator._make_category_normalizer(categories_map)
        self.kwargs = kwargs
        self.transform = None

    @override
    def _load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        tags_path = self._download(
            tags=self.tags_path
        )
        # CSV読み込み & 正規化
        return TagLoader.load_csv(tags_path,self.normalizer)

    # -------------------------
    # load
    # -------------------------
    @override
    def _load_model(self):
        model_path, preproc_path = self._download(
            model=self.model_path,
            preproc=self.preproc_path
        )

        #Preprocessパラメータ読み込み
        with open(preproc_path, "r") as f:
            preproc = json.load(f)       
        size, mean, std = self._load_preprocess_params(preproc["stages"])

        # ONNXロード
        return ModelFactory.load(
            model_path,
            param=ModelParam(
                color_format="RGB",
                use_normalize=True,
                normalize_mean=mean,
                normalize_std=std,
            )
        )

    # -------------------------
    # preprocess (stages only)
    # -------------------------
    def _load_preprocess_params(
            self,
            stages
    ) -> tuple[int, List[int], List[int]]:
        for s in stages:
            t = s["type"]
            if t == "resize":
                size = s["size"][0]
            elif t == "normalize":
                mean = s["mean"]
                std = s["std"]

        return size, mean, std

    @override
    def _interrogate_core(
        self,
        processed_image: np.ndarray,
        mask: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        # ONNX推論
        return self.model(output_name='prediction', **{self.model.input_names[0]: processed_image})

# =========================
# Camie Interrogator
# =========================
class CamieInterrogator(Interrogator):
    def __init__(
        self,
        name,
        model_path="model.onnx",
        metadata_path="metadata.json",
        **kwargs
    ):
        super().__init__(name, use_sigmoid=True)
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.kwargs = kwargs

    @override
    def _load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        # JSONからタグマッピングを読み込み
        # Load tag mapping from JSON        
        metadata_path = self._download(
            tags=self.metadata_path
        )

        schema = {
            "iterator": lambda d: d["dataset_info"]["tag_mapping"]["idx_to_tag"].items(),
            "id": lambda item: int(item[0]),
            "name": lambda item: item[1],
            "category": lambda item, d: d["dataset_info"]["tag_mapping"]["tag_to_category"].get(item[1], "general"),
        }

        #jsonからタグ情報読み込み
        return TagLoader.load_json(
            metadata_path,
            schema
        )

    @override
    def _load_model(self):
        model_path = self._download(
            model=self.model_path
        )

        # ONNXロード
        return ModelFactory.load(
            model_path,
            param=ModelParam(
                pad_color=(124, 116, 104)
            )
        )

    @override
    def _interrogate_core(
        self,
        processed_image: np.ndarray,
        mask: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        # ONNX推論
        return self.model(output_name='refined_predictions', **{self.model.input_names[0]: processed_image})

# =========================
# CL Interrogator
# =========================
class CLInterrogator(Interrogator):
    def __init__(
        self,
        name,
        model_path="model.onnx",
        metadata_path="tag_mapping.json",
        **kwargs
    ):
        super().__init__(name, use_sigmoid=True)
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.kwargs = kwargs

    @override
    def _load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        metadata_path = self._download(
            tags=self.metadata_path
        )

        schema = {
            "iterator": lambda d: d.items(),
            "id": lambda item: int(item[0]),
            "name": lambda item: item[1]["tag"],
            "category": lambda item: item[1]["category"],
        }

        #jsonからタグ情報読み込み
        return TagLoader.load_json(
            metadata_path,
            schema
        )

    @override
    def _load_model(self) -> None:
        model_path = self._download(
            model=self.model_path
        )

        # ONNXロード
        return ModelFactory.load(
            model_path,
            param=ModelParam(
                color_format="BGR"
            )
        )

    @override
    def _interrogate_core(
        self,
        processed_image: np.ndarray,
        mask: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        # ONNX推論
        return self.model( **{self.model.input_names[0]: processed_image})

# =========================
# General Interrogator
# 汎用ONNXタグ推論クラス
# =========================
class GeneralInterrogator(Interrogator):
    def __init__(
        self,
        name,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        use_normalize=True,
        normalize={"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]},
        color_format="RGB",
        categories_map={"0":"general","4":"character","9":"rating"},
        output_sigmoid=False,
        **kwargs
    ):
        super().__init__(name, use_sigmoid=output_sigmoid)
        self.model_path = model_path
        self.tags_path = tags_path
        self.use_normalize = use_normalize
        self.normalize = normalize
        self.color_format = color_format
        self.output_sigmoid=output_sigmoid
        self.normalizer = Interrogator._make_category_normalizer(categories_map)
        self.kwargs = kwargs

    @override
    def _load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        tags_path = self._download(
            tags=self.tags_path
        )
        # CSV読み込み & 正規化
        return TagLoader.load_csv(tags_path,self.normalizer)

    @override
    def _load_model(self):
        model_path = self._download(
            model=self.model_path
        )

        # ONNXロード
        return ModelFactory.load(
            model_path,
            param=ModelParam(
                color_format=self.color_format,
                use_normalize=self.use_normalize,
                normalize_mean=self.normalize["mean"] if self.normalize is not None else None,
                normalize_std=self.normalize["std"] if self.normalize is not None else None
            )
        )

    @override
    def _interrogate_core(
        self,
        processed_image: np.ndarray,
        mask: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        # ONNX推論
        return self.model( **{self.model.input_names[0]: processed_image})

# =========================
# General TIMM Interrogator
# timmベースモデル対応
# =========================
class GeneralTimmInterrogator(Interrogator):
    def __init__(
        self,
        name,
        model_path='model.safetensors',
        tags_path='selected_tags.csv',
        config_path='config.json',
        categories_map={"0":"general","4":"character","9":"rating"},
        **kwargs):
        super().__init__(name, use_sigmoid=True)
        self.model_path = model_path
        self.tags_path = tags_path
        self.config_path = config_path
        self.normalizer = Interrogator._make_category_normalizer(categories_map)
        self.kwargs = kwargs

    @override
    def _load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        tags_path = self._download(
            tags=self.tags_path
        )
        # CSV読み込み & 正規化
        return TagLoader.load_csv(tags_path,self.normalizer)

    @override
    def _load_model(self):
        model_path, config_path = self._download(
            model=self.model_path,
            config=self.config_path
        )

        # Timmロード
        return ModelFactory.load(
            model_path,
            config_path=config_path
        )

    @override
    def _interrogate_core(
        self,
        processed_image: np.ndarray,
        mask: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        # Timm推論
        return self.model( image=processed_image)
