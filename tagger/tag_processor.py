from dataclasses import dataclass
import re
from typing import Dict, List

from .interrogator import TagData

@dataclass
class TagResult:
    tags: Dict[str, float]          #ratingとCharactersを除いたタグ
    ratings: Dict[str, float]
    characters: Dict[str, float]

class TagProcessor:
    _escape_pattern = re.compile(r'([\\()])')

    @staticmethod
    def tag_grouping(tag_datas: List[TagData]):
        
        ratings = {}
        tags = {}
        characters = {}

        # 確率順にソート
        tag_datas.sort(key=lambda x: x.score, reverse=True)

        # 振り分け
        for tagdata in tag_datas:
            if tagdata.category == "rating":
                ratings[tagdata.name] = tagdata.score
            elif tagdata.category == "character":
                characters[tagdata.name] = tagdata.score
            else:
                tags[tagdata.name] = tagdata.score

        return TagResult(tags=tags, ratings=ratings, characters=characters)

    @staticmethod
    def _tag_threshold(
        tags: Dict[str, float],
        threshold=0.35,
        exclude_tags: List[str] | None = None,
        sort_by_alphabetical_order=False
    ) -> Dict[str, float]:

        exclude_tags = exclude_tags or []

        return {
            t: c
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )
            if c >= threshold and t not in exclude_tags
        }

    @staticmethod
    def postprocess_tags(
        tags: Dict[str, float],
        characters: Dict[str, float],
        threshold=0.35,
        character_threshold=0.35,
        additional_tags: List[str] | None = None,
        exclude_tags: List[str] | None = None,
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: List[str] | None = None,
        escape_tag=False
    ) -> Dict[str, float]:

        # 安全化（mutable対策）
        additional_tags = additional_tags or []
        exclude_tags = exclude_tags or []
        replace_underscore_excludes = replace_underscore_excludes or []

        # 強制追加タグ
        for t in additional_tags:
            tags[t] = 1.0

        # 閾値適用
        characters = TagProcessor._tag_threshold(
            characters, character_threshold, exclude_tags, sort_by_alphabetical_order
        )
        tags = TagProcessor._tag_threshold(
            tags, threshold, exclude_tags, sort_by_alphabetical_order
        )

        # マージ（仕様：characters優先）
        all_tags = {**tags, **characters}

        new_tags = []
        for tag in list(all_tags):
            new_tag = tag

            # underscore変換
            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            # escape
            if escape_tag:
                new_tag = TagProcessor._escape_pattern.sub(r'\\\1', new_tag)

            # weight付与
            if add_confident_as_weight:
                new_tag = f'({new_tag}:{all_tags[tag]})'

            new_tags.append((new_tag, all_tags[tag]))

        return dict(new_tags)