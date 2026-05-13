import os
import gc
import pandas as pd
import numpy as np

from collections import defaultdict
from typing import Optional,Tuple, List, Dict
from io import BytesIO
from PIL import Image

from pathlib import Path
from huggingface_hub import hf_hub_download


from modules import shared
from modules.deepbooru import re_special as tag_escape_pattern

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
    def postprocess_tags(
        tags: Dict[str, float],

        threshold=0.35,
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
        tags = {
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

        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)

        return tags

    def __init__(self, name: str) -> None:
        self.name = name

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


class DeepDanbooruInterrogator(Interrogator):
    def __init__(self, name: str, project_path: os.PathLike) -> None:
        super().__init__(name)
        self.project_path = project_path

    def load(self) -> None:
        print(f'Loading {self.name} from {str(self.project_path)}')

        # deepdanbooru package is not include in web-sd anymore
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/c81d440d876dfd2ab3560410f37442ef56fc663
        from launch import is_installed, run_pip
        if not is_installed('deepdanbooru'):
            package = os.environ.get(
                'DEEPDANBOORU_PACKAGE',
                'git+https://github.com/KichangKim/DeepDanbooru.git@d91a2963bf87c6a770d74894667e9ffa9f6de7ff'
            )

            run_pip(
                f'install {package} tensorflow tensorflow-io', 'deepdanbooru')

        import tensorflow as tf

        # tensorflow maps nearly all vram by default, so we limit this
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        # TODO: only run on the first run
        for device in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        with tf.device(tf_device_name):
            import deepdanbooru.project as ddp

            self.model = ddp.load_model_from_project(
                project_path=self.project_path,
                compile_model=False
            )

            print(f'Loaded {self.name} model from {str(self.project_path)}')

            self.tags = ddp.load_tags_from_project(
                project_path=self.project_path
            )

    def unload(self) -> bool:
        # unloaded = super().unload()

        # if unloaded:
        #     # tensorflow suck
        #     # https://github.com/keras-team/keras/issues/2102
        #     import tensorflow as tf
        #     tf.keras.backend.clear_session()
        #     gc.collect()

        # return unloaded

        # There is a bug in Keras where it is not possible to release a model that has been loaded into memory.
        # Downgrading to keras==2.1.6 may solve the issue, but it may cause compatibility issues with other packages.
        # Using subprocess to create a new process may also solve the problem, but it can be too complex (like Automatic1111 did).
        # It seems that for now, the best option is to keep the model in memory, as most users use the Waifu Diffusion model with onnx.

        return False

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

        import deepdanbooru.data as ddd

        # convert an image to fit the model
        image_bufs = BytesIO()
        image.save(image_bufs, format='PNG')
        image = ddd.load_image_for_evaluate(
            image_bufs,
            self.model.input_shape[2],
            self.model.input_shape[1]
        )

        image = image.reshape((1, *image.shape[0:3]))

        # evaluate model
        result = self.model.predict(image)

        confidents = result[0].tolist()
        ratings = {}
        tags = {}

        for i, tag in enumerate(self.tags):
            tags[tag] = confidents[i]

        return ratings, tags


class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
        self,
        name: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        **kwargs
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.kwargs = kwargs

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")

        model_path = Path(hf_hub_download(
            **self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(
            **self.kwargs, filename=self.tags_path))
        return model_path, tags_path

    def load(self) -> None:
        model_path, tags_path = self.download()

        # only one of these packages should be installed at a time in any one environment
        # https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime
        # TODO: remove old package when the environment changes?
        from launch import is_installed, run_pip
        if not is_installed('onnxruntime'):
            package = os.environ.get(
                'ONNXRUNTIME_PACKAGE',
                'onnxruntime-gpu'
            )

            run_pip(f'install {package}', 'onnxruntime')

        from onnxruntime import InferenceSession

        # https://onnxruntime.ai/docs/execution-providers/
        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/commit/e4ec460122cf674bbf984df30cdb10b4370c1224#r92654958
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_cpu:
            providers.pop(0)

        self.model = InferenceSession(str(model_path), providers=providers)

        print(f'Loaded {self.name} model from {model_path}')

        self.tags = pd.read_csv(tags_path)

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

        # code for converting the image and running the model is taken from the link below
        # thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

        # convert an image to fit the model
        _, height, _, _ = self.model.get_inputs()[0].shape
        
        print(f'{self.model.get_inputs()[0].shape}')

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidents = self.model.run([label_name], {input_name: image})[0]

        tags = self.tags[:][['name']]
        tags['confidents'] = confidents[0]

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)

        return ratings, tags

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
        self.kwargs = kwargs
        
    def download(self) -> Tuple[os.PathLike, os.PathLike,  Optional[os.PathLike]]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")

        model_path = Path(hf_hub_download(
            **self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(
            **self.kwargs, filename=self.tags_path))
        preproc_path = Path(hf_hub_download(
            **self.kwargs, filename=self.preproc_path))
        return model_path, tags_path, preproc_path
        
    def load(self):
        from onnxruntime import InferenceSession
        import json
        import numpy as np
        import pandas as pd

        model_path, tags_path, preproc_path = self.download()

        from launch import is_installed, run_pip
        if not is_installed('onnxruntime'):
            package = os.environ.get(
                'ONNXRUNTIME_PACKAGE',
                'onnxruntime-gpu'
            )

            run_pip(f'install {package}', 'onnxruntime')

        from onnxruntime import InferenceSession

        # https://onnxruntime.ai/docs/execution-providers/
        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/commit/e4ec460122cf674bbf984df30cdb10b4370c1224#r92654958
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_cpu:
            providers.pop(0)


        # ----------------------------
        # 3. load ONNX
        # ----------------------------
        self.model = InferenceSession(str(model_path), providers=providers)
        print(f'Loaded {self.name} model from {model_path}')
        # ----------------------------
        # 4. tags
        # ----------------------------
        self.tags_df = pd.read_csv(tags_path)

        # ----------------------------
        # 5. preprocessing
        # ----------------------------
        with open(preproc_path, "r", encoding="utf-8") as f:
            self.preproc = json.load(f)

        self.image_size = int(self.preproc["image_size"])
        self.pad_color = tuple(self.preproc["pad_color_rgb"])

        self.mean = np.array(self.preproc["normalize_mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(self.preproc["normalize_std"], dtype=np.float32).reshape(3, 1, 1)

        return True

    def preprocess(self, image):
        import numpy as np
        from PIL import Image

        image = image.convert("RGB")
        w, h = image.size
        size = self.image_size

        scale = min(size / w, size / h)
        nw, nh = int(w * scale), int(h * scale)

        image = image.resize((nw, nh), Image.BICUBIC)

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

    def interrogate(self, image, threshold=0.35,**kwargs):
        if not hasattr(self, "model"):
                self.load()

        import numpy as np

        pixel_values, padding_mask = self.preprocess(image)

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

        df = self.tags_df.copy()
        df["confidence"] = probs

        # 上位のみ
        df = df.sort_values("confidence", ascending=False)

        ratings = {
            k.replace("rating:", "", 1).strip(): v
            for k, v in zip(df["name"], df["confidence"])
            if str(k).startswith("rating:")
        }

        # rating: を含まないものだけ tags に残す
        tags = {
            k: v
            for k, v in zip(df["name"], df["confidence"])
            if not str(k).startswith("rating:")
        }
        return ratings, tags

class PixAIInterrogator(Interrogator):
    def __init__(
    self,
    name,
    model_path='model.onnx',
    tags_path='selected_tags.csv',
    preproc_path='preprocess.json',
    character_thresholds=None,
    **kwargs):
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.preproc_path = preproc_path
        self.character_thresholds = character_thresholds
        self.kwargs = kwargs

        self.model = None
        self.tags_df = None
        self.transform = None

    def download(self) -> Tuple[os.PathLike, os.PathLike, Optional[os.PathLike]]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")

        model_path = Path(hf_hub_download(
            **self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(
            **self.kwargs, filename=self.tags_path))
        if self.preproc_path is None:
            preproc_path = None
        else:
            preproc_path = Path(hf_hub_download(
                **self.kwargs, filename=self.preproc_path))
        return model_path, tags_path, preproc_path

    # -------------------------
    # load
    # -------------------------
    def load(self):
        from onnxruntime import InferenceSession
        import json
        import numpy as np
        import pandas as pd

        model_path, tags_path, preproc_path = self.download()

        from launch import is_installed, run_pip
        if not is_installed('onnxruntime'):
            package = os.environ.get(
                'ONNXRUNTIME_PACKAGE',
                'onnxruntime-gpu'
            )

            run_pip(f'install {package}', 'onnxruntime')

        from onnxruntime import InferenceSession

        # https://onnxruntime.ai/docs/execution-providers/
        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/commit/e4ec460122cf674bbf984df30cdb10b4370c1224#r92654958
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_cpu:
            providers.pop(0)
        
        model_path, tags_path, preproc_path = self.download()
        
        self.model = InferenceSession(str(model_path) , providers=providers)
        self.tags_df = pd.read_csv(tags_path)
        # IPS構造復元（PixAI仕様）
        self.d_ips = {}
        if 'ips' in self.tags_df.columns:
            self.tags_df['ips'] = self.tags_df['ips'].map(json.loads)
            for name, ips in zip(self.tags_df['name'], self.tags_df['ips']):
                if ips:
                    self.d_ips[name] = ips

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
                    x = x.resize((size, size), Image.BILINEAR)

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
    def interrogate(self, image,
        threshold=0.35,**kwargs):
        if self.model is None:
            self.load()

        character_thresholds = kwargs.get("character_thresholds", self.character_thresholds)
        if character_thresholds is None:
            character_thresholds = 0
        else:
            character_thresholds = float(character_thresholds)

        values = self._predict(image)

        prediction = values["prediction"]

        df = self.tags_df.copy()
        tags = {}
        general = {}
        character = {}

        # --- category分離
        for category in sorted(df["category"].unique()):
            mask = df["category"] == category
            tag_names = df["name"][mask].tolist()
            category_pred = prediction[mask]

            cat_tags = {}
            # category名取得（PixAI仕様）
            cat_name = str(category)

            values_key = values.get("category_names", {})
            if values_key:
                cat_name = values_key.get(category, cat_name)

            for name, score in zip(tag_names, category_pred):
                if cat_name == "4" and float(score) >= character_thresholds:
                    cat_tags[name] = float(score)
                elif cat_name == "0":
                    cat_tags[name] = float(score)

            if cat_name == "0":
                general = cat_tags
            elif cat_name == "4":
                character = cat_tags

        # general + character を統合
        tags.update(general)
        tags.update(character)

        # -------------------------
        # IPS処理
        # -------------------------
        ips_mapping = {}
        ips_counts = defaultdict(int)

        for tag, score in character.items():
            if float(score) >= threshold:
                if tag in self.d_ips:
                    ips_mapping[tag] = self.d_ips[tag]
                    for ip in self.d_ips[tag]:
                        ips_counts[ip] += 1

        ips = sorted(ips_counts.items(), key=lambda x: (-x[1], x[0]))
        ips = [x[0] for x in ips]

        ratings = {}
        # score=1.0固定でratingsへ
        for ip, _ in ips_counts.items():
            ratings[ip] = 1.0
        return ratings, tags

class CamieInterrogator(Interrogator):
    def __init__(
        self,
        name,
        model_path="model.onnx",
        metadata_path="metadata.json",
        category_thresholds=None,
        min_confidence=0.1,
        fmt=("rating", "general", "character"),
        **kwargs
    ):
        super().__init__(name)
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.category_thresholds = category_thresholds
        self.min_confidence = min_confidence
        self.fmt = fmt
        self.kwargs = kwargs

    def download(self):
        from huggingface_hub import hf_hub_download
        from pathlib import Path

        model_path = Path(hf_hub_download(
            **self.kwargs,
            filename=self.model_path
        ))

        metadata_path = Path(hf_hub_download(
            **self.kwargs,
            filename=self.metadata_path
        ))

        return model_path, metadata_path

    def load(self):
        import json
        import onnxruntime as ort

        model_path, metadata_path = self.download()

        # --- metadata ---
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.metadata = metadata

        # --- providers自動選択 ---
        providers = []
        try:
            if ort.get_device() == "GPU":
                providers.append("CUDAExecutionProvider")
        except Exception:
            pass
        providers.append("CPUExecutionProvider")

        # --- session ---
        self.model = ort.InferenceSession(
            str(model_path),
            providers=providers
        )

        self.input_name = self.model.get_inputs()[0].name

        # --- tag mapping ---
        if 'dataset_info' in metadata:
            mapping = metadata['dataset_info']['tag_mapping']
            self.idx_to_tag = mapping['idx_to_tag']
            self.tag_to_category = mapping['tag_to_category']
        else:
            self.idx_to_tag = metadata.get('idx_to_tag', {})
            self.tag_to_category = metadata.get('tag_to_category', {})

        self.image_size = metadata.get('model_info', {}).get('img_size', 512)

        return True

    # --- preprocess ---
    def preprocess(self, image):
        import numpy as np
        from PIL import Image
        import torchvision.transforms as transforms

        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')

        w, h = image.size
        aspect = w / h

        if aspect > 1:
            new_w = self.image_size
            new_h = int(new_w / aspect)
        else:
            new_h = self.image_size
            new_w = int(new_h * aspect)

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        pad_color = (124, 116, 104)
        canvas = Image.new('RGB', (self.image_size, self.image_size), pad_color)

        paste_x = (self.image_size - new_w) // 2
        paste_y = (self.image_size - new_h) // 2
        canvas.paste(image, (paste_x, paste_y))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transform(canvas).numpy()

    def interrogate(self, image, threshold=0.35,**kwargs):
        import numpy as np

        if not hasattr(self, "model"):
            self.load()
        
        fmt = kwargs.get("fmt", self.fmt)
        
        category_thresholds = kwargs.get("category_thresholds", self.category_thresholds)

        # --- preprocess ---
        img = self.preprocess(image)
        img = np.expand_dims(img, 0)

        # --- inference ---
        outputs = self.model.run(None, {self.input_name: img})

        # --- refined優先 ---
        logits = outputs[1] if len(outputs) >= 2 else outputs[0]

        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = probs[0]

        # --- 整理 ---
        all_probs = {}
        for idx, prob in enumerate(probs):
            prob = float(prob)
            if prob < self.min_confidence:
                continue

            tag = self.idx_to_tag.get(str(idx), f"unknown-{idx}")
            cat = self.tag_to_category.get(tag, "general")

            if cat not in all_probs:
                all_probs[cat] = []

            all_probs[cat].append((tag, prob))

        # sort
        for cat in all_probs:
            all_probs[cat].sort(key=lambda x: x[1], reverse=True)

        # threshold適用
        selected = {}
        for cat, items in all_probs.items():
            if category_thresholds and cat in category_thresholds:
                th = category_thresholds[cat]
            else:
                th = 0

            selected[cat] = [(t, p) for t, p in items if p >= th]

        # --- fmtフィルタ ---
        selected = {k: v for k, v in selected.items() if k in fmt}

        # --- 出力整形 ---
        ratings = dict(selected.get("rating", []))

        tags = {
            tag: prob
            for cat, items in selected.items()
            if cat != "rating"
            for tag, prob in items
        }

        return ratings, tags


class GeneralInterrogator(Interrogator):
    def __init__(
        self,
        name,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        doStd=True,
        normalize={"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]},
        rgb="RGB",
        **kwargs):
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.doStd = doStd
        self.normalize = normalize
        self.rgb = rgb
        self.kwargs = kwargs

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")

        model_path = Path(hf_hub_download(
            **self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(
            **self.kwargs, filename=self.tags_path))
        return model_path, tags_path
        
    def load(self):
        from onnxruntime import InferenceSession
        import json
        import numpy as np
        import pandas as pd

        model_path, tags_path = self.download()

        from launch import is_installed, run_pip
        if not is_installed('onnxruntime'):
            package = os.environ.get(
                'ONNXRUNTIME_PACKAGE',
                'onnxruntime-gpu'
            )

            run_pip(f'install {package}', 'onnxruntime')

        from onnxruntime import InferenceSession

        # https://onnxruntime.ai/docs/execution-providers/
        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/commit/e4ec460122cf674bbf984df30cdb10b4370c1224#r92654958
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_cpu:
            providers.pop(0)

        # ----------------------------
        # 3. load ONNX
        # ----------------------------
        self.model = InferenceSession(str(model_path), providers=providers)
        print(f'Loaded {self.name} model from {model_path}')
        # ----------------------------
        # 4. tags
        # ----------------------------
        self.tags_df = pd.read_csv(tags_path)

        return True
    def infer_layout(self, shape):
        if len(shape) != 4:
            return "NCHW"

        # チャンネル候補（よくある値）
        channel_candidates = [1, 3, 4]

        if shape[1] in channel_candidates:
            return "NCHW"
        elif shape[-1] in channel_candidates:
            return "NHWC"
        return "NCHW"

    def interrogate(self, image, threshold=0.35,**kwargs):
        if not hasattr(self, "model"):
                self.load()

        import numpy as np
        shape = self.model.get_inputs()[0].shape
        #レイアウト判定
        layout = self.infer_layout(shape)
        if layout == "NHWC" :
            _, height, wwidth, c = shape
        if layout == "NCHW" :
            _, c, height, wwidth = shape


        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        if self.rgb == "BGR":
            image = image[:, :, ::-1]  # BGR

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)

        image = image.astype(np.float32)
        #正規化
        if self.doStd:
            image = image / 255.0
       
        if layout == "NCHW":
            image = image.transpose(2, 0, 1)
            if self.normalize is not None:
                mean = np.array(self.normalize["mean"], dtype=np.float32).reshape(3, 1, 1)
                std = np.array(self.normalize["std"], dtype=np.float32).reshape(3, 1, 1)
                image = (image - mean) / std
        else:
            if self.normalize is not None:
                mean = np.array(self.normalize["mean"], dtype=np.float32).reshape(1, 1, 3)
                std = np.array(self.normalize["std"], dtype=np.float32).reshape(1, 1, 3)
                image = (image - mean) / std


        image = np.expand_dims(image, 0)

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        
        # フラット化
        confidents = np.asarray(
            self.model.run([output_name], {input_name: image})[0]
        ).reshape(-1)

        tags = {
           str(name): float(score)
           for name, score in zip(self.tags_df["name"], confidents)
        }
        ratings={}

        return ratings, tags
