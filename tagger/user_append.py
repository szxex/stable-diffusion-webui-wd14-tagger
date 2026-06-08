from typing import Dict

from tagger.interrogator import Interrogator\
    , MLDanbooruInterrogator\
    , WaifuDiffusionInterrogator\
    , OracleInterrogator\
    , PixAIInterrogator\
    , CamieInterrogator\
    , CLInterrogator\
    , GeneralInterrogator\
    , GeneralTimmInterrogator

append_interrogators: Dict[str, Interrogator] = {
      
        #'safetensor-timm': GeneralTimmInterrogator(
        #    'safetensor-timm',
        #     repo_id='repo_id',
        #),
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
