import os

from typing import List, Dict
from pathlib import Path

from modules import shared, scripts
from preload import default_ddp_path
from tagger.preset import Preset
from tagger.interrogator import Interrogator, DeepDanbooruInterrogator, WaifuDiffusionInterrogator, TransformerInterrogator, PixAIInterrogator, GeneralInterrogator

preset = Preset(Path(scripts.basedir(), 'presets'))

interrogators: Dict[str, Interrogator] = {}


def refresh_interrogators() -> List[str]:
    global interrogators
    interrogators = {
        'wd-convnext-v3': WaifuDiffusionInterrogator(
            'wd14-convnext-v3-git',
            repo_id='SmilingWolf/wd-convnext-tagger-v3'
        ),
        'wd-vit-v3': WaifuDiffusionInterrogator(
            'wd14-vit-v3',
            repo_id='SmilingWolf/wd-vit-tagger-v3'
        ),
        'wd-swinv2-v3': WaifuDiffusionInterrogator(
            'wd-swinv2-v3',
            repo_id='SmilingWolf/wd-swinv2-tagger-v3'
        ),
        'wd-ViT-Large-v3': WaifuDiffusionInterrogator(
            'WD ViT-Large-v3',
            repo_id='SmilingWolf/wd-vit-large-tagger-v3',
        ),
        'wd-EVA02-Large-v3': WaifuDiffusionInterrogator(
            'wd-EVA02-Large-v3',
             repo_id='SmilingWolf/wd-eva02-large-tagger-v3',
        ),
        'idolsankaku-eva02-large-v1': WaifuDiffusionInterrogator(
            'idolsankaku-eva02-large-v1',
             repo_id='deepghs/idolsankaku-eva02-large-tagger-v1',
        ),
        'idolsankaku-swinv2-v1': WaifuDiffusionInterrogator(
            'idolsankaku-swinv2-v1',
             repo_id='deepghs/idolsankaku-swinv2-tagger-v1',
        ),
        'pixai-tagger-v0.9': PixAIInterrogator(
            'pixai-tagger-v0.9',
             repo_id='deepghs/pixai-tagger-v0.9-onnx',
        ),
        'pixai-tagger-v0.9E': GeneralInterrogator(
            'pixai-tagger-v0.9E',
             repo_id='etset/pixai-tagger-v0.9E',
             model_path='pixai-tagger-v0.9.onnx',
             tags_path='pixai-tagger-v0.9.csv',
             key_name='output',
        ),
        'OppaiOracle-v1.1': TransformerInterrogator(
            'OppaiOracle-v1.1',
             model_path='V1.1_onnx/model.onnx',
             tags_path='V1.1_onnx/selected_tags.csv',
             preproc_path='V1.1_onnx/preprocessing.json',
             repo_id='Grio43/OppaiOracle',
             
        ),
    }

    # load deepdanbooru project
    deepdanbooru_projects_path = 'models/torch_deepdanbooru'
    shared.cmd_opts.deepdanbooru_projects_path = deepdanbooru_projects_path
    os.makedirs(
        getattr(shared.cmd_opts, 'deepdanbooru_projects_path', default_ddp_path),
        exist_ok=True
    )

    for path in os.scandir(shared.cmd_opts.deepdanbooru_projects_path):
        if not path.is_dir():
            continue

        if not Path(path, 'project.json').is_file():
            continue

        interrogators[path.name] = DeepDanbooruInterrogator(path.name, path)

    return sorted(interrogators.keys())
def split_str(s: str, separator=',') -> List[str]:
    return [x.strip() for x in s.split(separator) if x]
