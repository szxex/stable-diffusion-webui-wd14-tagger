import os
import json
import gradio as gr
import ast

from collections import OrderedDict
from pathlib import Path
from glob import glob
from PIL import Image, UnidentifiedImageError

from modules.call_queue import wrap_gradio_gpu_call
from modules import ui
try:
    from modules import generation_parameters_copypaste as parameters_copypaste
except ImportError:
    from modules import infotext_utils as parameters_copypaste 

from tagger import format, utils
from tagger.utils import split_str
from tagger.interrogator import Interrogator


def unload_interrogators():
    unloaded_models = 0

    for i in utils.interrogators.values():
        if i.unload():
            unloaded_models = unloaded_models + 1

    return [f'Successfully unload {unloaded_models} model(s)']

def parse_kwargs(kwargs_str: str) -> dict:
    """
    キーワード引数形式の文字列を辞書(dict)に変換する。
    空文字列の場合は空の辞書を返す。
    """
    # 空文字列や空白のみの場合は空の辞書を返す
    if not kwargs_str or not kwargs_str.strip():
        return {}
    
    # ダミーの関数呼び出し文字列を作成して構文解析（パース）する
    # 例: "fmt=('a'), aa=1" -> "dummy(fmt=('a'), aa=1)"
    code = f"dummy({kwargs_str})"
    
    try:
        # 文字列をPythonの構文木（AST）に変換
        tree = ast.parse(code)
        
        # 関数呼び出し部分のノードを取得
        call_node = tree.body[0].value
        
        # キーワード引数を抽出して辞書に格納
        kwargs_dict = {}
        for keyword in call_node.keywords:
            if keyword.arg is not None:
                # ast.literal_eval で文字列からPythonのデータ型へ安全に変換
                kwargs_dict[keyword.arg] = ast.literal_eval(keyword.value)
                
        return kwargs_dict

    except SyntaxError as e:
        raise ValueError(f"構文エラーです。キーワード引数の形式を確認してください: {kwargs_str}") from e
    except ValueError as e:
        raise ValueError(f"安全に評価できない値が含まれています: {kwargs_str}") from e

# カテゴリのリストを受け取り、カテゴリごとの閾値を文字列として返す関数
def categories_to_text(
        categories: list,
        threshold: float,
        character_threshold: float
    ) -> str:
    if not categories:
        return ""
    threshold_map = {
        "character": character_threshold,
    }

    # カテゴリごとの閾値
    cat_threshold = ', '.join(
        f'"{c}":{threshold_map.get(c, threshold)}'
        for c in sorted(categories)
    )

    return f"category_thresholds={{{cat_threshold}}}"

#モデルのカテゴリ読み込み
def load_interrogator_categories(
    prev_selected: list,
    interrogator: str,
    prev_interrogator: str
):
    if interrogator not in utils.interrogators:
        return (gr.update(choices=[], value=[],interactive=True), interrogator)

    interrogator_model: Interrogator = utils.interrogators[interrogator]

    categories,tags = interrogator_model.load_tags()
    categories = sorted(list(categories))
    if interrogator != prev_interrogator:
        #If the interrogator has been changed, reset the selected categories to all categories of the new interrogator.
        new_value = categories
    else:
        new_value = [c for c in prev_selected if c in categories]
    if (
        interrogator == prev_interrogator and
        set(prev_selected) == set(new_value)
    ):
        # 何も変わってないので再描画しない
        return (gr.update(), interrogator)

    return (gr.update(choices=categories, value=new_value,interactive=True), interrogator)

def on_interrogate(
    image: Image,
    batch_input_glob: str,
    batch_input_recursive: bool,
    batch_output_dir: str,
    batch_output_filename_format: str,
    batch_output_action_on_conflict: str,
    batch_remove_duplicated_tag: bool,
    batch_output_save_json: bool,

    interrogator: str,
    interrogator_option: str,
    threshold: float,
    character_threshold: float,
    categories_checkbox: list,
    additional_tags: str,
    exclude_tags: str,
    sort_by_alphabetical_order: bool,
    add_confident_as_weight: bool,
    replace_underscore: bool,
    replace_underscore_excludes: str,
    escape_tag: bool,
    display_max_tags: int,

    unload_model_after_running: bool
):
    if interrogator not in utils.interrogators:
        return ['', None, None, None, f"'{interrogator}' is not a valid interrogator"]

    interrogator: Interrogator = utils.interrogators[interrogator]

    categories = set(categories_checkbox)

    process_opts = (
    )
    process_kwargs = {
        **parse_kwargs(interrogator_option),
        "categories": categories
    }

    postprocess_opts = (
        threshold,
        character_threshold,
        split_str(additional_tags),
        split_str(exclude_tags),
        sort_by_alphabetical_order,
        add_confident_as_weight,
        replace_underscore,
        split_str(replace_underscore_excludes),
        escape_tag
    )

    # single process
    if image is not None:
        categories, ratings, tags, characters = interrogator.interrogate(image,*process_opts,**process_kwargs)
        processed_tags = Interrogator.postprocess_tags(
            tags,
            characters,
            *postprocess_opts
        )

        if unload_model_after_running:
            interrogator.unload()
     
     
        return [
            ', '.join(processed_tags),
            ratings,
            dict(list(tags.items())[:display_max_tags]),
            dict(list(characters.items())[:10]),
            ''
        ]

    # batch process
    batch_input_glob = batch_input_glob.strip()
    batch_output_dir = batch_output_dir.strip()
    batch_output_filename_format = batch_output_filename_format.strip()

    if batch_input_glob != '':
        # if there is no glob pattern, insert it automatically
        if not batch_input_glob.endswith('*'):
            if not batch_input_glob.endswith(os.sep):
                batch_input_glob += os.sep
            batch_input_glob += '*'

        # get root directory of input glob pattern
        base_dir = batch_input_glob.replace('?', '*')
        base_dir = base_dir.split(os.sep + '*').pop(0)

        # check the input directory path
        if not os.path.isdir(base_dir):
            return ['', None, None, None, 'input path is not a directory']

        # this line is moved here because some reason
        # PIL.Image.registered_extensions() returns only PNG if you call too early
        supported_extensions = [
            e
            for e, f in Image.registered_extensions().items()
            if f in Image.OPEN
        ]

        paths = [
            Path(p)
            for p in glob(batch_input_glob, recursive=batch_input_recursive)
            if '.' + p.split('.').pop().lower() in supported_extensions
        ]

        print(f'found {len(paths)} image(s)')

        for path in paths:
            try:
                image = Image.open(path)
            except UnidentifiedImageError:
                # just in case, user has mysterious file...
                print(f'${path} is not supported image type')
                continue

            # guess the output path
            base_dir_last = Path(base_dir).parts[-1]
            base_dir_last_idx = path.parts.index(base_dir_last)
            output_dir = Path(
                batch_output_dir) if batch_output_dir else Path(base_dir)
            output_dir = output_dir.joinpath(
                *path.parts[base_dir_last_idx + 1:]).parent

            output_dir.mkdir(0o777, True, True)

            # format output filename
            format_info = format.Info(path, 'txt')

            try:
                formatted_output_filename = format.pattern.sub(
                    lambda m: format.format(m, format_info),
                    batch_output_filename_format
                )
            except (TypeError, ValueError) as error:
                return ['', None, None, str(error)]

            output_path = output_dir.joinpath(
                formatted_output_filename
            )

            output = []

            if output_path.is_file():
                output.append(output_path.read_text(errors='ignore').strip())

                if batch_output_action_on_conflict == 'ignore':
                    print(f'skipping {path}')
                    continue

            categories, ratings, tags, characters = interrogator.interrogate(image,*process_opts,**process_kwargs)
            processed_tags = Interrogator.postprocess_tags(
                tags,
                characters,
                *postprocess_opts
            )

            # TODO: switch for less print
            print(
                f'found {len(processed_tags)} tags out of {len(tags)} from {path}'
            )

            plain_tags = ', '.join(processed_tags)

            if batch_output_action_on_conflict == 'copy':
                output = [plain_tags]
            elif batch_output_action_on_conflict == 'prepend':
                output.insert(0, plain_tags)
            else:
                output.append(plain_tags)

            if batch_remove_duplicated_tag:
                output_path.write_text(
                    ', '.join(
                        OrderedDict.fromkeys(
                            map(str.strip, ','.join(output).split(','))
                        )
                    ),
                    encoding='utf-8'
                )
            else:
                output_path.write_text(
                    ', '.join(output),
                    encoding='utf-8'
                )

            if batch_output_save_json:
                output_path.with_suffix('.json').write_text(
                    json.dumps([ratings, characters, tags])
                )

        print('all done :)')

    if unload_model_after_running:
        interrogator.unload()

    return ['', None, None, None, '']

def RowCompat(*args, **kwargs):
    row = gr.Row(*args)

    # 古いgradioにstyleがある場合
    if hasattr(row, "style"):
        return row.style(**kwargs)
    
    # 新しいgradioはそのまま
    return gr.Row(*args, **kwargs)

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        #state
        prev_interrogator_state = gr.State("")
        with RowCompat(equal_height=False):
            with gr.Column(variant='panel'):

                # input components
                with gr.Tabs():
                    with gr.TabItem(label='Single process'):
                        image = gr.Image(
                            label='Source',
                            source='upload',
                            interactive=True,
                            type="pil"
                        )

                    with gr.TabItem(label='Batch from directory'):
                        batch_input_glob = utils.preset.component(
                            gr.Textbox,
                            label='Input directory',
                            placeholder='/path/to/images or /path/to/images/**/*'
                        )
                        batch_input_recursive = utils.preset.component(
                            gr.Checkbox,
                            label='Use recursive with glob pattern'
                        )

                        batch_output_dir = utils.preset.component(
                            gr.Textbox,
                            label='Output directory',
                            placeholder='Leave blank to save images to the same path.'
                        )

                        batch_output_filename_format = utils.preset.component(
                            gr.Textbox,
                            label='Output filename format',
                            placeholder='Leave blank to use same filename as original.',
                            value='[name].[output_extension]'
                        )

                        import hashlib
                        with gr.Accordion(
                            label='Output filename formats',
                            open=False
                        ):
                            gr.Markdown(
                                value=f'''
                                ### Related to original file
                                - `[name]`: Original filename without extension
                                - `[extension]`: Original extension
                                - `[hash:<algorithms>]`: Original extension
                                    Available algorithms: `{', '.join(hashlib.algorithms_available)}`

                                ### Related to output file
                                - `[output_extension]`: Output extension (has no dot)

                                ## Examples
                                ### Original filename without extension
                                `[name].[output_extension]`

                                ### Original file's hash (good for deleting duplication)
                                `[hash:sha1].[output_extension]`
                                '''
                            )

                        batch_output_action_on_conflict = utils.preset.component(
                            gr.Dropdown,
                            label='Action on existing caption',
                            value='ignore',
                            choices=[
                                'ignore',
                                'copy',
                                'append',
                                'prepend'
                            ]
                        )

                        batch_remove_duplicated_tag = utils.preset.component(
                            gr.Checkbox,
                            label='Remove duplicated tag'
                        )

                        batch_output_save_json = utils.preset.component(
                            gr.Checkbox,
                            label='Save with JSON'
                        )

                submit = gr.Button(
                    value='Interrogate',
                    variant='primary'
                )

                info = gr.HTML()

                # preset selector
                with gr.Row(variant='compact'):
                    available_presets = utils.preset.list()
                    selected_preset = gr.Dropdown(
                        label='Preset',
                        choices=available_presets,
                        value=available_presets[0]
                    )

                    save_preset_button = gr.Button(
                        value=ui.save_style_symbol
                    )

                    ui.create_refresh_button(
                        selected_preset,
                        lambda: None,
                        lambda: {'choices': utils.preset.list()},
                        'refresh_preset'
                    )

                # option components

                # interrogator selector
                with gr.Column():
                    with gr.Row(variant='compact'):
                        interrogator_names = utils.refresh_interrogators()
                        interrogator = utils.preset.component(
                            gr.Dropdown,
                            label='Interrogator',
                            choices=interrogator_names,
                            value=(
                                None
                                if len(interrogator_names) < 1 else
                                interrogator_names[-1]
                            )
                        )

                        ui.create_refresh_button(
                            interrogator,
                            lambda: None,
                            lambda: {'choices': utils.refresh_interrogators()},
                            'refresh_interrogator'
                        )
                        
                        interrogator_option = utils.preset.component(
                            gr.Textbox,
                            label='Interrogator Option (split by comma)',
                            elem_id='interrogator-option'
                        )

                    unload_all_models = gr.Button(
                        value='Unload all interrogate models'
                    )

                threshold = utils.preset.component(
                    gr.Slider,
                    label='Threshold',
                    minimum=0,
                    maximum=1,
                    value=0.35
                )
                character_threshold = utils.preset.component(
                    gr.Slider,
                    label='Character Threshold',
                    minimum=0,
                    maximum=1,
                    value=0.35
                )

                with gr.Accordion(
                    label='Categories',
                    open=False
                ):
                    load_categories = gr.Button(
                        value='Load Categories'
                    )
                    categories_checkbox = gr.CheckboxGroup(label="Categories")
                    categories_text = gr.Textbox(
                        label="Categories (Threshold)",
                        interactive=False
                    )

                additional_tags = utils.preset.component(
                    gr.Textbox,
                    label='Additional tags (split by comma)',
                    elem_id='additioanl-tags'
                )

                exclude_tags = utils.preset.component(
                    gr.Textbox,
                    label='Exclude tags (split by comma)',
                    elem_id='exclude-tags'
                )

                sort_by_alphabetical_order = utils.preset.component(
                    gr.Checkbox,
                    label='Sort by alphabetical order',
                )
                add_confident_as_weight = utils.preset.component(
                    gr.Checkbox,
                    label='Include confident of tags matches in results'
                )
                replace_underscore = utils.preset.component(
                    gr.Checkbox,
                    label='Use spaces instead of underscore',
                    value=True
                )
                replace_underscore_excludes = utils.preset.component(
                    gr.Textbox,
                    label='Excudes (split by comma)',
                    # kaomoji from WD 1.4 tagger csv. thanks, Meow-San#5400!
                    value='0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||'
                )
                escape_tag = utils.preset.component(
                    gr.Checkbox,
                    label='Escape brackets',
                )

                unload_model_after_running = utils.preset.component(
                    gr.Checkbox,
                    label='Unload model after running',
                )

                display_max_tags = utils.preset.component(
                    gr.Number,
                    label='Display tags',
                    elem_id='display_max_tags',
                    value=1000,
                    precision=0
                )

            # output components
            with gr.Column(variant='panel'):
                tags = gr.Textbox(
                    label='Tags',
                    placeholder='Found tags',
                    interactive=False
                )

                with gr.Row():
                    parameters_copypaste.bind_buttons(
                        parameters_copypaste.create_buttons(
                            ["txt2img", "img2img"],
                        ),
                        None,
                        tags
                    )

                rating_confidents = gr.Label(
                    label='Rating confidents',
                    elem_id='rating-confidents'
                )
                character_confidents = gr.Label(
                    label='Character confidents',
                    elem_id='character-confidents'
                )
                tag_confidents = gr.Label(
                    label='Tag confidents',
                    elem_id='tag-confidents'
                )


        # register events
        selected_preset.change(
            fn=utils.preset.apply,
            inputs=[selected_preset],
            outputs=[*utils.preset.components, info]
        )

        save_preset_button.click(
            fn=utils.preset.save,
            inputs=[selected_preset, *utils.preset.components],  # values only
            outputs=[info]
        )

        unload_all_models.click(
            fn=unload_interrogators,
            outputs=[info]
        )
        
        load_categories.click(
            fn=load_interrogator_categories,
            inputs=[categories_checkbox,interrogator,prev_interrogator_state],
            outputs=[categories_checkbox,prev_interrogator_state]
        )

        for func in [categories_checkbox.change, threshold.change, character_threshold.change]:
            func(
                fn=categories_to_text,
                inputs=[categories_checkbox, threshold, character_threshold],
                outputs=[categories_text]
            )

        for func in [image.change, submit.click]:
            
            event = func(
                fn=load_interrogator_categories,
                inputs=[categories_checkbox,interrogator,prev_interrogator_state],
                outputs=[categories_checkbox,prev_interrogator_state]
            )
            event.then(
                fn=wrap_gradio_gpu_call(on_interrogate),
                inputs=[
                    # single process
                    image,

                    # batch process
                    batch_input_glob,
                    batch_input_recursive,
                    batch_output_dir,
                    batch_output_filename_format,
                    batch_output_action_on_conflict,
                    batch_remove_duplicated_tag,
                    batch_output_save_json,

                    # options
                    interrogator,
                    interrogator_option,
                    threshold,
                    character_threshold,
                    categories_checkbox,
                    additional_tags,
                    exclude_tags,
                    sort_by_alphabetical_order,
                    add_confident_as_weight,
                    replace_underscore,
                    replace_underscore_excludes,
                    escape_tag,
                    display_max_tags,

                    unload_model_after_running
                ],
                outputs=[
                    tags,
                    rating_confidents,
                    tag_confidents,
                    character_confidents,
                    # contains execution time, memory usage and other stuffs...
                    # generated from modules.ui.wrap_gradio_call
                    info
                ]
            )


    return [(tagger_interface, "Tagger", "tagger")]
