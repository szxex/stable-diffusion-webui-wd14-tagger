import os
import gc
import re
import pandas as pd
import numpy as np
import json

from collections import defaultdict
from typing import Optional,Tuple, List, Dict,Callable, Set
try:
    from typing import override
except ImportError:
    from typing_extensions import override
from io import BytesIO
from PIL import Image

from pathlib import Path
from huggingface_hub import hf_hub_download

from modules import shared

# タグ内の特殊文字「\」「(」「)」をエスケープする正規表現
# Regex to escape special characters "\" "(" ")"
tag_escape_pattern = re.compile(r'([\\()])')

from . import dbimutils

# -------------------------
# デバイス選択（CPU / GPU）
# Select execution device (CPU or GPU)
# -------------------------
use_cpu = ('all' in shared.cmd_opts.use_cpu) or (
    'interrogate' in shared.cmd_opts.use_cpu)

if use_cpu:
    tf_device_name = '/cpu:0'
else:
    tf_device_name = '/gpu:0'

    # GPU IDが指定されている場合
    # If specific GPU device is given
    if shared.cmd_opts.device_id is not None:
        try:
            tf_device_name = f'/gpu:{int(shared.cmd_opts.device_id)}'
        except ValueError:
            print('--device-id is not a integer')


# =========================
# Base Interrogator Class
# 共通ベースクラス
# =========================
class Interrogator:

    def __init__(self, name: str) -> None:
        self.name = name
        self.model_categories = ('general')

    # -------------------------
    # タグの閾値フィルタリング
    # Filter tags by confidence threshold
    # -------------------------
    @staticmethod
    def _tag_threshold(
        tags: Dict[str, float],
        threshold=0.35,
        exclude_tags: List[str] = [],
        sort_by_alphabetical_order=False
    ) -> Dict[str, float]:
        return {
            t: c

            # 並び替え（タグ名 or 信頼度）
            # Sorting (by name or confidence)
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )

            # フィルタ条件
            # Filtering condition
            if (
                c >= threshold
                and t not in exclude_tags
            )
        }    


    # -------------------------
    # タグの後処理
    # Tag postprocessing
    # -------------------------
    @staticmethod
    def postprocess_tags(
        tags: Dict[str, float],
        characters: Dict[str, float],
        threshold=0.35,
        character_threshold=0.35,
        additional_tags: List[str] = [],
        exclude_tags: List[str] = [],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: List[str] = [],
        escape_tag=False
    ) -> Dict[str, float]:

        # 強制追加タグ
        # Force-add tags
        for t in additional_tags:
            tags[t] = 1.0

        # カテゴリごとに閾値適用
        # Apply thresholds separately
        characters = Interrogator._tag_threshold(characters,character_threshold,exclude_tags,sort_by_alphabetical_order)
        tags = Interrogator._tag_threshold(tags,threshold,exclude_tags,sort_by_alphabetical_order)

        # マージ
        # Merge dicts
        all_tags =characters | tags

        new_tags = []
        for tag in list(all_tags):
            new_tag = tag

            # "_" → " " に変換（除外指定あり）
            # Replace underscores with spaces
            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            # エスケープ
            # Escape special chars
            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            # weight付加形式 "(tag:score)"
            # Add confidence as weight format
            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, all_tags[tag]))

        tags = dict(new_tags)
        return tags


    # -------------------------
    # ONNXモデルロード
    # Load ONNX model with provider selection
    # -------------------------
    def _load_onnx(self, model_path):
        from launch import is_installed, run_pip

        # onnxruntime未インストールなら自動インストール
        if not is_installed('onnxruntime'):
            package = os.environ.get('ONNXRUNTIME_PACKAGE', 'onnxruntime-gpu')
            run_pip(f'install {package}', 'onnxruntime')

        from onnxruntime import InferenceSession

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_cpu:
            providers = ['CPUExecutionProvider']

        return InferenceSession(str(model_path), providers=providers)

    # -------------------------
    # HuggingFaceからファイルDL
    # Download files from HuggingFace
    # -------------------------
    def _download(self, **files) -> List[Path] | Path:
        paths = [
            Path(hf_hub_download(**self.kwargs, filename=f))
            for f in files.values()
        ]
        return paths[0] if len(paths) == 1 else paths

    # -------------------------
    # 入力テンソルのレイアウト推測
    # Detect tensor layout (NCHW or NHWC)
    # -------------------------
    @staticmethod
    def _infer_layout(shape) -> str:
        if len(shape) != 4:
            return "NCHW"

        # よく使用されるチャンネル数
        channel_candidates = [1, 3, 4]

        if shape[1] in channel_candidates:
            return "NCHW"
        elif shape[-1] in channel_candidates:
            return "NHWC"
        return "NCHW"
        
    # -------------------------
    # 汎用前処理（全モデル用）
    # Generic image preprocessing
    # -------------------------
    @staticmethod
    def _general_preproccess(
        model,
        image,
        rgb= "RGB",
        pad_color = (255, 255, 255),
        do_standard=True,
        normalize={"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}
    ) -> np.ndarray:
        import numpy as np
        
        shape = model.get_inputs()[0].shape
        
        # レイアウト判定
        layout = Interrogator._infer_layout(shape)
        if layout == "NHWC" :
            _, height, width, c = shape
        if layout == "NCHW" :
            _, c, height, width = shape

        # サイズ未定義なら448固定
        if height == 'height':
            image_size = 448
        else:
            image_size = height

        # RGBA → 背景白で合成
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')

        # アスペクト比維持リサイズ
        w, h = image.size
        aspect = w / h

        if aspect > 1:
            new_w = image_size
            new_h = int(new_w / aspect)
        else:
            new_h = image_size
            new_w = int(new_h * aspect)

        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # パディングして正方形化
        canvas = Image.new('RGB', (image_size, image_size), pad_color)

        paste_x = (image_size - new_w) // 2
        paste_y = (image_size - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))
        
        arr = np.asarray(canvas, dtype=np.float32)

        # BGR変換
        if rgb == "BGR":
            arr = arr[:, :, ::-1]

        # 0-1正規化
        if do_standard:
            arr /= 255.0

        # 転置（NCHW）
        if layout == "NCHW":
            arr = arr.transpose(2, 0, 1)
            if normalize is not None:
                mean = np.array(normalize["mean"], dtype=np.float32).reshape(3, 1, 1)
                std = np.array(normalize["std"], dtype=np.float32).reshape(3, 1, 1)
        else:
            if normalize is not None:
                mean = np.array(normalize["mean"], dtype=np.float32).reshape(1, 1, 3)
                std = np.array(normalize["std"], dtype=np.float32).reshape(1, 1, 3)

        # 正規化適用
        if normalize is not None:
            arr = (arr - mean) / std

        return np.expand_dims(arr.astype(np.float32), 0)

    @staticmethod
    def make_category_normalizer(categories_map) -> Callable[[str,str], str]:
        def _normalize(name, cat):
            return categories_map.get(cat, cat)
        return _normalize

    # -------------------------
    # CSV読み込み
    # Load tag CSV file
    # -------------------------
    @staticmethod
    def _load_csv(
        csv_path,
        _normalize_category: Optional[Callable[[str,str], str]] = None
    ) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        df = pd.read_csv(csv_path)

        # category列補正
        df["category"] = df.get("category", "general")
        df["category"] = df["category"].fillna("general").astype(str).str.lower()

        # カテゴリ正規化
        if _normalize_category:
            df["category"] = df.apply(
                lambda r: _normalize_category(r["name"], str(r["category"])),
                axis=1
            )

        df = df[["name", "category"]]
        categories = set(df["category"])
        result = df.to_dict(orient="index")

        return categories, result
    
    # -------------------------
    # シグモイド関数
    # sigmoid activation
    # -------------------------
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    def _category_allowed(self, cat: str, categories) -> bool:
        if categories is None:
            return True
        return cat in categories


    # 出力構築
    # Build structured output (rating / tags / characters)
    def _build_output(
            self,
            probs,
            categories=None,
            category_thresholds=None
        ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

        ratings = {}
        tags = {}
        caracters = {}
        items = []

        # すべてのタグに対して分類
        for idx, prob in enumerate(probs):
            prob = float(prob)

            tag_info = self.tags.get(idx)
            if tag_info is None:
                continue

            name = tag_info["name"]
            cat  = tag_info["category"]

            # カテゴリフィルタ
            if categories is not None and cat not in categories:
                continue

            # カテゴリ別閾値
            if category_thresholds and cat in category_thresholds:
                th = category_thresholds[cat]
            else:
                th = 0

            if prob < th or prob < 0.01:
                continue

            items.append((cat, name, prob))

        # 確率順にソート
        items.sort(key=lambda x: x[2], reverse=True)

        # 振り分け
        for cat, name, prob in items:
            if cat == "rating":
                ratings[name] = prob
            elif cat == "character":
                caracters[name] = prob
            else:
                tags[name] = prob

        return self.model_categories, ratings, tags, caracters

    def load_tags(self)-> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def unload(self) -> bool:
        unloaded = False

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            unloaded = True
            print(f'Unloaded {self.name}')

        if hasattr(self, 'tags'):
            del self.tags

        if hasattr(self, 'metadata'):
            del self.metadata
        return unloaded

    def interrogate(
        self,
        image: Image,
        **kwargs
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        raise NotImplementedError()

#WaifuDiffusionタグ付けモデル
class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
        self,
        name: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        categories_map={"0":"general","4":"character","9":"rating"},
        **kwargs
    ) -> None:
        # 親クラス初期化
        super().__init__(name)

        # モデルとタグのパス
        self.model_path = model_path
        self.tags_path = tags_path

        # 使用するカテゴリ（None = 全部）
        self.categories = None
        self.category_thresholds = None

        # カテゴリ番号 → 名前 の変換
        # Convert numeric category IDs into readable category names
        self.normalizer = Interrogator.make_category_normalizer(categories_map)

        # HF用引数
        self.kwargs = kwargs

    @override
    def load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        # 既に読み込み済みならスキップ
        if(hasattr(self, 'tags') and self.tags is not None):
            return self.model_categories, self.tags

        # HuggingFaceからCSV取得
        tags_path = self._download(tags=self.tags_path)

        # CSV読み込み & 正規化
        self.model_categories, self.tags = Interrogator._load_csv(tags_path,self.normalizer)

        return self.model_categories, self.tags

    @override
    def load(self):
        # 既にロード済みなら何もしない
        if hasattr(self, 'model') and self.model is not None:
            return

        # モデルDL
        model_path = self._download(model=self.model_path)

        # ONNXロード
        self.model = self._load_onnx(model_path)

        # タグ読み込み
        self.load_tags()

    @override
    def interrogate(self, image: Image, **kwargs):
        # モデル未ロードならロード
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        # 外部引数でカテゴリ制御
        categories = kwargs.get("categories", self.categories)
        category_thresholds = kwargs.get("category_thresholds", self.category_thresholds)

        # ✅ 前処理（BGR & 正規化なし）
        # WaifuDiffusionモデルはOpenCV系（BGR）入力前提
        image = self._general_preproccess(
            self.model,
            image,
            rgb="BGR",
            do_standard=False,
            normalize=None
        )

        # ONNX推論
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name

        probs = self.model.run([label_name], {input_name: image})[0][0]

        # 出力整形
        return self._build_output(probs,categories,category_thresholds)

#ML-Danbooruモデル
class MLDanbooruInterrogator(Interrogator):
    def __init__(
        self,
        name: str,
        model_path='ml_caformer_m36_dec-5-97527.onnx',
        tags_path='tags.csv',
        **kwargs
    ) -> None:
        super().__init__(name)

        self.model_path = model_path
        self.tags_path = tags_path

        self.categories = None
        self.category_thresholds = None

        self.normalizer = None
        self.kwargs = kwargs

    @override
    def load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        if(hasattr(self, 'tags') and self.tags is not None):
            return self.model_categories, self.tags

        tags_path = self._download(tags=self.tags_path)
        
        df = pd.read_csv(tags_path)

        # カラム補正
        # Normalize columns (some datasets differ)
        df["category"] = df.get("category", "general")
        df["name"] = df.get("tag")

        self.model_categories = set(df["category"])

        # index→tag情報辞書
        self.tags = df[["name", "category"]].to_dict(orient="index")

        return self.model_categories, self.tags

    @override
    def load(self):
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path = self._download(model=self.model_path)

        self.model = self._load_onnx(model_path)
        self.load_tags()

    @override
    def interrogate(self, image: Image, **kwargs):
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        categories = kwargs.get("categories", self.categories)
        category_thresholds = kwargs.get("category_thresholds", self.category_thresholds)

        # ✅ 標準正規化あり（0〜1）
        image = self._general_preproccess(
            self.model,
            image,
            do_standard=True,
            normalize=None
        )

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name

        output = self.model.run([label_name], {input_name: image})[0]

        # ✅ Sigmoid適用（このモデルはlogits出力）
        probs = Interrogator._sigmoid(output).flatten()

        return self._build_output(probs,categories,category_thresholds)

 
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
    def _normalize_category(name,cat):
        if name.lower().startswith("rating:"):
            return "rating"
        if cat == "0":
            return "general"
        return cat

    @override
    def load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        if(hasattr(self, 'tags') and self.tags is not None):
            return self.model_categories, self.tags

        tags_path = self._download(
            tags=self.tags_path
        )
        self.model_categories, self.tags =  Interrogator._load_csv(tags_path,self.normalizer)
        return self.model_categories, self.tags 

    @override
    def load(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path, preproc_path = self._download(
             model=self.model_path,
             preproc=self.preproc_path
        )

        self.model = self._load_onnx(model_path)
        self.load_tags()

        with open(preproc_path, "r") as f:
            self.preproc = json.load(f)

        self.image_size = int(self.preproc["image_size"])
        self.pad_color = tuple(self.preproc["pad_color_rgb"])

        self.mean = np.array(self.preproc["normalize_mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(self.preproc["normalize_std"], dtype=np.float32).reshape(3, 1, 1)

    def _preprocess(self, image):

        image = image.convert("RGB")
        w, h = image.size
        size = self.image_size

        scale = min(size / w, size / h)
        nw, nh = int(w * scale), int(h * scale)

        image = image.resize((nw, nh), Image.LANCZOS)

        canvas = Image.new("RGB", (size, size), self.pad_color)
        x0 = (size - nw) // 2
        y0 = (size - nh) // 2
        canvas.paste(image, (x0, y0))

        mask = np.ones((size, size), dtype=bool)
        mask[y0:y0+nh, x0:x0+nw] = False

        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        arr = (arr - self.mean) / self.std

        return arr.astype(np.float32), mask

    @override
    def interrogate(
        self,
        image: Image,
        **kwargs
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        pixel_values, padding_mask = self._preprocess(image)

        input_name = self.model.get_inputs()[0].name
        mask_name = self.model.get_inputs()[1].name
        output_name = self.model.get_outputs()[0].name

        probs = self.model.run(
                [output_name],
                {
                        input_name: pixel_values[None, ...],
                        mask_name: padding_mask[None, ...],
                },
        )[0][0]

        return self._build_output(probs,categories=None,category_thresholds=None)

class PixAIInterrogator(Interrogator):
    def __init__(
    self,
    name,
    model_path='model.onnx',
    tags_path='selected_tags.csv',
    preproc_path='preprocess.json',
    category_thresholds=None,
    categories=("general", "character"),
    categories_map={"0":"general","4":"character"},
    **kwargs):
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.preproc_path = preproc_path
        self.category_thresholds = category_thresholds
        self.categories = categories
        self.normalizer = Interrogator.make_category_normalizer(categories_map)
        self.kwargs = kwargs
        self.transform = None

    @override
    def load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        if(hasattr(self, 'tags') and self.tags is not None):
            return self.model_categories, self.tags

        tags_path = self._download(
            tags=self.tags_path
        )
        self.model_categories, self.tags =  Interrogator._load_csv(tags_path,self.normalizer)
        return self.model_categories, self.tags 

    # -------------------------
    # load
    # -------------------------
    @override
    def load(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path, preproc_path = self._download(
            model=self.model_path,
            preproc=self.preproc_path
        )

        self.model = self._load_onnx(model_path)
        self.load_tags()

        if self.preproc_path is None:
            self.transform = None
        else:
            with open(preproc_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.transform = self._build_transform(cfg["stages"])

    # -------------------------
    # preprocess (stages only)
    # -------------------------
    def _build_transform(self, stages):
        def pipeline(image):
            x = image.convert("RGB")

            for s in stages:
                t = s["type"]

                if t == "resize":
                    size = s["size"][0]
                    x = x.resize((size, size), Image.LANCZOS)

                elif t == "to_tensor":
                    x = np.array(x).astype(np.float32) / 255.0
                    x = x.transpose(2, 0, 1)

                elif t == "normalize":
                    mean = np.array(s["mean"], dtype=np.float32).reshape(3, 1, 1)
                    std = np.array(s["std"], dtype=np.float32).reshape(3, 1, 1)
                    x = (x - mean) / std

            return x.astype(np.float32)

        return pipeline

    # -------------------------
    # inference
    # -------------------------
    def _predict(self, image):
        if self.transform is not None:
            x = self.transform(image)[None, ...]
        else:
            # ✅ preprocessなし時（最低限の形にする）
            import numpy as np
            image = image.convert("RGB").resize((448, 448), Image.BILINEAR)
            x = np.array(image).astype(np.float32)
            x = x.transpose(2, 0, 1)[None, ...] / 255.0
        input_name = self.model.get_inputs()[0].name
        output_names = [o.name for o in self.model.get_outputs()]

        outputs = self.model.run(output_names, {input_name: x})

        return {name: value[0] for name, value in zip(output_names, outputs)}

    # -------------------------
    # wd14-compatible API
    # -------------------------
    def interrogate(
        self,
        image: Image,
        **kwargs
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        #Additional Parameter
        categories = kwargs.get("categories", self.categories)
        category_thresholds = kwargs.get("category_thresholds", self.category_thresholds)

        values = self._predict(image)

        probs = values["prediction"]

        return self._build_output(probs,categories,category_thresholds)


class CamieInterrogator(Interrogator):
    def __init__(
        self,
        name,
        model_path="model.onnx",
        metadata_path="metadata.json",
        category_thresholds=None,
        categories=("rating", "general", "character"),
        **kwargs
    ):
        super().__init__(name)
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.category_thresholds = category_thresholds
        self.categories = categories
        self.kwargs = kwargs

    @override
    def load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        if(hasattr(self, 'tags') and self.tags is not None):
            return self.model_categories, self.tags

        metadata_path = self._download(
            tags=self.metadata_path
        )
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        mapping = metadata.get("dataset_info", {}).get("tag_mapping", {})

        self.tags = {
            int(idx): {
                "name": tag,
                "category": mapping['tag_to_category'].get(tag, "general")
            }
            for idx, tag in mapping['idx_to_tag'].items()
        }
        self.model_categories = {v["category"] for v in self.tags.values()}

        self.image_size = metadata.get("model_info", {}).get("img_size", 512)

        return self.model_categories, self.tags

    @override
    def load(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path = self._download(
            model=self.model_path
        )

        self.model = self._load_onnx(model_path)
        self.load_tags()

    @override
    def interrogate(
        self,
        image: Image,
        **kwargs
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        #Additonal Parameter
        categories = kwargs.get("categories", self.categories)
        category_thresholds = kwargs.get("category_thresholds", self.category_thresholds)

        # --- preprocess ---
        img = self._general_preproccess(
            self.model,
            image,
            pad_color = (124, 116, 104)
        )

        # --- inference ---
        self.input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {self.input_name: img})

        # --- refined優先 ---
        logits = outputs[1] if len(outputs) >= 2 else outputs[0]

        probs = Interrogator._sigmoid(logits).flatten()

        return self._build_output(probs,categories,category_thresholds)

class CLInterrogator(Interrogator):
    def __init__(
        self,
        name,
        model_path="model.onnx",
        metadata_path="tag_mapping.json",
        category_thresholds=None,
        categories=("rating", "general", "character"),
        **kwargs
    ):
        super().__init__(name)
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.category_thresholds = category_thresholds
        self.categories = categories
        self.kwargs = kwargs

    @override
    def load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        if(hasattr(self, 'tags') and self.tags is not None):
            return self.model_categories, self.tags

        metadata_path = self._download(
            tags=self.metadata_path
        )
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.tags = {
            int(idx): {
                "name": v["tag"],
                "category": v["category"].lower()
            }
            for idx, v in metadata.items()
        }
        self.model_categories = {v["category"] for v in self.tags.values()}
        return self.model_categories, self.tags 

    @override
    def load(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path = self._download(
            model=self.model_path
        )

        self.model = self._load_onnx(model_path)
        self.load_tags()

    @override
    def interrogate(
        self,
        image: Image,
        **kwargs
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()
        
        categories = kwargs.get("categories", self.categories)
        category_thresholds = kwargs.get("category_thresholds", self.category_thresholds)

        #Image Preprocess
        image = self._general_preproccess(
            self.model,
            image
        )
        
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        # --- inference ---
        outputs = self.model.run([self.output_name], {self.input_name: image})

        if np.isnan(outputs[0]).any() or np.isinf(outputs[0]).any():
            outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = Interrogator._sigmoid(outputs[0]).flatten()
        
        return self._build_output(probs,categories,category_thresholds)

class GeneralInterrogator(Interrogator):
    def __init__(
        self,
        name,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        do_standard=True,
        normalize={"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]},
        rgb="RGB",
        categories=None,
        category_thresholds=None,
        categories_map={"0":"general","4":"character","9":"rating"},
        output_sigmoid=False,
        **kwargs):
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.do_standard = do_standard
        self.normalize = normalize
        self.rgb = rgb
        self.categories=categories
        self.category_thresholds=category_thresholds
        self.output_sigmoid=output_sigmoid
        self.normalizer = Interrogator.make_category_normalizer(categories_map)
        self.kwargs = kwargs

    @override
    def load_tags(self) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        if(hasattr(self, 'tags') and self.tags is not None):
            return self.model_categories, self.tags

        tags_path = self._download(
            tags=self.tags_path
        )
        self.model_categories, self.tags =  Interrogator._load_csv(tags_path,self.normalizer)
        return self.model_categories, self.tags 

    @override
    def load(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path = self._download(
            model=self.model_path
        )

        self.model = self._load_onnx(model_path)
        self.load_tags()

    @override
    def interrogate(
        self,
        image: Image,
        **kwargs
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()
        #Additonal Parameter
        categories = kwargs.get("categories", self.categories)
        category_thresholds = kwargs.get("category_thresholds", self.category_thresholds)
        
        #Image Preprocess
        image = self._general_preproccess(
            self.model,
            image,
            rgb=self.rgb,
            do_standard=self.do_standard,
            normalize=self.normalize
        )

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        
        # 実行
        probs = self.model.run([output_name], {input_name: image})

        # sigmoidが必要なモデルとそうでないモデルがあるため、オプションで切り替えられるようにする
        if self.output_sigmoid:
            probs = Interrogator._sigmoid(probs[0])
        else:
            probs = probs[0]
        probs = probs.flatten()

        return self._build_output(probs,categories,category_thresholds)
