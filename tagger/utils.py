import os


from typing import List, Dict
from pathlib import Path

from modules import shared, scripts
from preload import default_ddp_path
from tagger.preset import Preset
from tagger.interrogator import Interrogator, DeepDanbooruInterrogator, WaifuDiffusionInterrogator, OracleInterrogator, PixAIInterrogator,CamieInterrogator, GeneralInterrogator

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
             #character_thresholds=0.6
             #defalt character_thresholds=None *character_thresholds=0
        ),
        'pixai-tagger-v0.9E': GeneralInterrogator(
            'pixai-tagger-v0.9E',
             repo_id='etset/pixai-tagger-v0.9E',
             model_path='pixai-tagger-v0.9.onnx',
             tags_path='pixai-tagger-v0.9.csv',
        ),
        'OppaiOracle-v1.1': OracleInterrogator(
            'OppaiOracle-v1.1',
             model_path='V1.1_onnx/model.onnx',
             tags_path='V1.1_onnx/selected_tags.csv',
             preproc_path='V1.1_onnx/preprocessing.json',
             repo_id='Grio43/OppaiOracle',
        ),
        'camie-tagger-v2': CamieInterrogator(
            'camie-tagger-v2',
             model_path='camie-tagger-v2.onnx',
             metadata_path='camie-tagger-v2-metadata.json',
             repo_id='Camais03/camie-tagger-v2',
             #fmt=('artist','character','copyright','general','meta','rating','year') 
             #default fmt = ('character','general','rating') 
             #category_thresholds={"artist":0.5,"character":0.8,"copyright":0.7,"general":0.3,"meta":0.2,"rating":0.5,"year":0.2} 
             #default category_thresholds=None *same thresholds
        ),
        #'tagger-name': GeneralInterrogator(
        #    'tagger-name',
        #     repo_id='repo_id',
        #     model_path='model.onnx',
        #     tags_path='selected_tags.csv',
        #     doStd=True,
        #      *defalut is True,normarize 0-1
        #     normalize={"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}, 
        #      *If you don't want to normalize the data, specify `None`. The values may be listed in the model description, `preprocess.json`, `config.json`, or similar files.
        #     rgb="RGB",
        #      *default is "RGB",If the results look strange, setting it to “BGR” might fix the problem.
        #),
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
