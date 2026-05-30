import os


from typing import List, Dict
from pathlib import Path

from modules import shared, scripts
from tagger.preset import Preset
from tagger.interrogator import Interrogator\
    , MLDanbooruInterrogator\
    , WaifuDiffusionInterrogator\
    , OracleInterrogator\
    , PixAIInterrogator\
    , CamieInterrogator\
    , CLInterrogator\
    , GeneralInterrogator

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
             #categories=('artist','character','copyright','general','meta','rating','year') 
             #default categories = ('character','general','rating') 
             #category_thresholds={"artist":0.5,"character":0.8,"copyright":0.7,"general":0.3,"meta":0.2,"rating":0.5,"year":0.2} 
             #default category_thresholds=None *same thresholds
        ),
        'cl-tagger-v1.02': CLInterrogator(
            'cl-tagger-v1.02',
             model_path='cl_tagger_1_02/model_optimized.onnx',
             metadata_path='cl_tagger_1_02/tag_mapping.json',
             repo_id='cella110n/cl_tagger',
             #categories=('Artist','Character','Copyright','General','Meta','Rating','Quality','Model') 
             #default categories = ('Character','General','Rating') 
             #category_thresholds={"Artist":0.5,"Character":0.8,"Copyright":0.7,"General":0.3,"Meta":0.2,"Rating":0.5,"Quality":0.2,"Model":0.5} 
             #default category_thresholds=None *same thresholds
        ),
        'Z3D-E621-Convnext': GeneralInterrogator(
            'Z3D-E621-Convnext',
             repo_id='toynya/Z3D-E621-Convnext',
             tags_path='tags-selected.csv',
             categories_map={'0':'general','1':'artist','3':'copyright','4':'character','5':'species','7':'meta','8':'lore'},
             rgb='BGR',
             do_standard=False,
             normalize=None,
        ),        
        'ML-Danbooru-dec-5-97527': MLDanbooruInterrogator(
            'ML-Danbooru-dec-5-97527',
            model_path='ml_caformer_m36_dec-5-97527.onnx',
             repo_id='deepghs/ml-danbooru-onnx',
        ),
        'ML-Danbooru-dec-3-80000': MLDanbooruInterrogator(
            'ML-Danbooru-dec-3-80000',
            model_path='ml_caformer_m36_dec-3-80000.onnx',
             repo_id='deepghs/ml-danbooru-onnx',
        ),
        #'tagger-name': GeneralInterrogator(
        #    'tagger-name',
        #     repo_id='repo_id',
        #     model_path='model.onnx',
        #     tags_path='selected_tags.csv',
        #     do_standard=True,
        #      *defalut is True,normarize 0-1
        #     normalize={"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}, 
        #      *If you don't want to normalize the data, specify `None`. The values may be listed in the model description, `preprocess.json`, `config.json`, or similar files.
        #     rgb='RGB',
        #      *default is "RGB",If the results look strange, setting it to “BGR” might fix the problem.
        #),
    }

    return sorted(interrogators.keys())


def split_str(s: str, separator=',') -> List[str]:
    return [x.strip() for x in s.split(separator) if x]
