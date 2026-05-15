import os
import gc
import re
import pandas as pd
import numpy as np
import json

from collections import defaultdict
from typing import Optional,Tuple, List, Dict,Callable, Set
from io import BytesIO
from PIL import Image

from pathlib import Path
from huggingface_hub import hf_hub_download

from modules import shared

tag_escape_pattern = re.compile(r'([\\()])')

# i'm not sure if it's okay to add this file to the repository
from . import dbimutils

# select a device to process
use_cpu = ('all' in shared.cmd_opts.use_cpu) or (
    'interrogate' in shared.cmd_opts.use_cpu)

if use_cpu:
    tf_device_name = '/cpu:0'
else:
    tf_device_name = '/gpu:0'

    if shared.cmd_opts.device_id is not None:
        try:
            tf_device_name = f'/gpu:{int(shared.cmd_opts.device_id)}'
        except ValueError:
            print('--device-id is not a integer')


class Interrogator:

    @staticmethod
    def _tag_threshold(
        tags: Dict[str, float],
        threshold=0.35,
        exclude_tags: List[str] = [],
        sort_by_alphabetical_order=False
    ) -> Dict[str, float]:
        return {
            t: c

            # sort by tag name or confident
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )

            # filter tags
            if (
                c >= threshold
                and t not in exclude_tags
            )
        }    


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
        for t in additional_tags:
            tags[t] = 1.0

        # those lines are totally not "pythonic" but looks better to me
        characters = Interrogator._tag_threshold(characters,character_threshold,exclude_tags,sort_by_alphabetical_order)
        tags = Interrogator._tag_threshold(tags,threshold,exclude_tags,sort_by_alphabetical_order)

        all_tags =characters | tags

        new_tags = []
        for tag in list(all_tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, all_tags[tag]))
        tags = dict(new_tags)

        return tags

    def __init__(self, name: str) -> None:
        self.name = name
        self.model_categories = ('general')

    def _load_onnx(self, model_path):
        from launch import is_installed, run_pip

        if not is_installed('onnxruntime'):
            package = os.environ.get('ONNXRUNTIME_PACKAGE', 'onnxruntime-gpu')
            run_pip(f'install {package}', 'onnxruntime')

        from onnxruntime import InferenceSession

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_cpu:
            providers = ['CPUExecutionProvider']

        return InferenceSession(str(model_path), providers=providers)

    def _download(self, **files):
        return [
            Path(hf_hub_download(**self.kwargs, filename=f))
            for f in files.values()
        ]

    @staticmethod
    def _infer_layout(shape):
        if len(shape) != 4:
            return "NCHW"

        # チャンネル候補（よくある値）
        channel_candidates = [1, 3, 4]

        if shape[1] in channel_candidates:
            return "NCHW"
        elif shape[-1] in channel_candidates:
            return "NHWC"
        return "NCHW"
        
    # -------------------------
    # unified preprocessing
    # -------------------------
    @staticmethod
    def _general_preproccess(
        model,
        image,
        rgb= "RGB",
        pad_color = (255, 255, 255),
        do_standard=True,
        normalize={"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}
    ):
        import numpy as np
        
        shape = model.get_inputs()[0].shape
        
        #レイアウト判定
        layout = Interrogator._infer_layout(shape)
        if layout == "NHWC" :
            _, height, width, c = shape
        if layout == "NCHW" :
            _, c, height, width = shape

        #モデルのインプットがサイズを返してくれない場合は固定値
        if height == 'height':
            image_size = 448
        else:
            image_size = height

        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')

        w, h = image.size
        aspect = w / h

        if aspect > 1:
            new_w = image_size
            new_h = int(new_w / aspect)
        else:
            new_h = image_size
            new_w = int(new_h * aspect)

        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        canvas = Image.new('RGB', (image_size, image_size), pad_color)

        paste_x = (image_size - new_w) // 2
        paste_y = (image_size - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))
        
        arr = np.asarray(canvas, dtype=np.float32)

        if rgb == "BGR":
            arr = arr[:, :, ::-1]

        if do_standard:
            arr /= 255.0

        if layout == "NCHW":
            arr = arr.transpose(2, 0, 1)
            if normalize is not None:
                mean = np.array(normalize["mean"], dtype=np.float32).reshape(3, 1, 1)
                std = np.array(normalize["std"], dtype=np.float32).reshape(3, 1, 1)
        else:
            if normalize is not None:
                mean = np.array(normalize["mean"], dtype=np.float32).reshape(1, 1, 3)
                std = np.array(normalize["std"], dtype=np.float32).reshape(1, 1, 3)

        if normalize is not None:
            arr = (arr - mean) / std

        return np.expand_dims(arr.astype(np.float32), 0)

    @staticmethod
    def make_category_normalizer(categories_map):
        def _normalize(name, cat):
            return categories_map.get(cat, cat)
        return _normalize

    @staticmethod
    def _load_csv(
        csv_path,
        _normalize_category: Optional[Callable[[str,str], str]] = None
    ) -> Tuple[Set[str], Dict[int, Dict[str, str]]]:
        df = pd.read_csv(csv_path)
        #補間
        df["category"] = df.get("category", "general")
        df["category"] = df["category"].fillna("general").astype(str).str.lower()

        if _normalize_category:
            df["category"] = df.apply(
                lambda r: _normalize_category(r["name"], str(r["category"])),
                axis=1
            )
        df = df[["name", "category"]]
        categories = set(df["category"])
        result = df.to_dict(orient="index")
        return categories, result

    def _category_allowed(cat: str, categories):
        if categories is None:
            return True
        return cat in categories

    def _build_output(self, probs,categories=None, category_thresholds=None):
        ratings = {}
        tags = {}
        caracters = {}
        items = []

        for idx, prob in enumerate(probs):
            prob = float(prob)

            tag_info = self.tags.get(idx)
            if tag_info is None:
                continue

            name = tag_info["name"]
            cat  = tag_info["category"]

            if categories is not None and cat not in categories:
                continue

            # category threshold適用
            if category_thresholds and cat in category_thresholds:
                th = category_thresholds[cat]
            else:
                th = 0

            if prob < th or prob < 0.01:
                continue

            items.append((cat, name, prob))

        items.sort(key=lambda x: x[2], reverse=True)  # prob順

        for cat, name, prob in items:
            if cat == "rating":
                ratings[name] = prob
            elif cat == "character":
                caracters[name] = prob
            else:
                tags[name] = prob
        return self.model_categories, ratings, tags, caracters

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
        threshold=0.35,
        **kwargs
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        raise NotImplementedError()


class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
        self,
        name: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        categories_map={"0":"general","4":"character","9":"rating"},
        **kwargs
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.categories = None
        self.category_thresholds = None
        self.normalizer = Interrogator.make_category_normalizer(categories_map)
        self.kwargs = kwargs

    def load(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path, tags_path = self._download(
            model=self.model_path,
            tags=self.tags_path
        )

        self.model = self._load_onnx(model_path)
        self.model_categories, self.tags = Interrogator._load_csv(tags_path,self.normalizer)

    def interrogate(
        self,
        image: Image,
        threshold=0.35,
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

        image = self._general_preproccess(
            self.model,
            image,
            rgb="BGR",
            do_standard=False,
            normalize=None
        )

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: image})[0][0]

        return self._build_output(probs,categories,category_thresholds)

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

    def load(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path, tags_path = self._download(
            model=self.model_path,
            tags=self.tags_path
        )

        self.model = self._load_onnx(model_path)
        df = pd.read_csv(tags_path)
        df["category"] = df.get("category", "general")
        df["name"] = df.get("tag")

        self.model_categories = set(df["category"])
        self.tags = df[["name", "category"]].to_dict(orient="index")

    def interrogate(
        self,
        image: Image,
        threshold=0.35,
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

        image = self._general_preproccess(
            self.model,
            image,
            do_standard=True,
            normalize=None
        )

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        output = self.model.run([label_name], {input_name: image})[0]
        probs = (1 / (1 + np.exp(-output))).reshape(-1)

        print(f'tags={self.tags}')
        print(f'output={probs}')

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

    def _normalize_category(name,cat):
        if name.lower().startswith("rating:"):
            return "rating"
        if cat == "0":
            return "general"
        return cat

    def load(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path, tags_path, preproc_path = self._download(
             model=self.model_path,
             tags=self.tags_path,
             preproc=self.preproc_path
        )

        self.model = self._load_onnx(model_path)
        self.model_categories, self.tags = Interrogator._load_csv(tags_path,self.normalizer)

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

    def interrogate(
        self,
        image: Image,
        threshold=0.35,
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

    # -------------------------
    # load
    # -------------------------
    def load(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path, tags_path, preproc_path = self._download(
            model=self.model_path,
            tags=self.tags_path,
            preproc=self.preproc_path
        )

        self.model = self._load_onnx(model_path)

        self.model_categories, self.tags = Interrogator._load_csv(tags_path,self.normalizer)

        # IPS構造復元（PixAI仕様）
        #self.d_ips = {}
        #if 'ips' in self.tags.columns:
        #    self.tags['ips'] = self.tags['ips'].map(json.loads)
        #    for name, ips in zip(self.tags['name'], self.tags['ips']):
        #       if ips:
        #            self.d_ips[name] = ips

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
        threshold=0.35,
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

    def load(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path, metadata_path = self._download(
            model=self.model_path,
            meta=self.metadata_path
        )

        self.model = self._load_onnx(model_path)

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

    def interrogate(
        self,
        image: Image,
        threshold=0.35,
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

        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = probs[0]

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

    def load(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path, metadata_path = self._download(
            model=self.model_path,
            meta=self.metadata_path
        )

        self.model = self._load_onnx(model_path)

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

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    def interrogate(
        self,
        image: Image,
        threshold=0.35,
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
        probs = self._sigmoid(outputs[0]).flatten()
        
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
        **kwargs):
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.do_standard = do_standard
        self.normalize = normalize
        self.rgb = rgb
        self.categories=None
        self.category_thresholds=None
        self.normalizer = Interrogator.make_category_normalizer(categories_map)
        self.kwargs = kwargs

    def load(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            return

        model_path, tags_path = self._download(
            model=self.model_path,
            tags=self.tags_path
        )

        self.model = self._load_onnx(model_path)
        self.model_categories, self.tags = Interrogator._load_csv(tags_path,self.normalizer)

    def interrogate(
        self,
        image: Image,
        threshold=0.35,
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
        
        # フラット化
        probs = self.model.run([output_name], {input_name: image})[0][0]

        return self._build_output(probs,categories,category_thresholds)
