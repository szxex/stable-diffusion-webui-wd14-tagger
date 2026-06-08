import inspect
import json

import pandas as pd
from typing import Tuple, Dict, Set, Callable

class TagLoader:

    #CSV読み込み。tagsは"name","category"に揃える
    @staticmethod
    def load_csv(
        path,
        normalize: Callable[[str, str], str] | None = None,
        tag_name: str = "name",
        category_name: str = "category"
    ) -> Tuple[Set[str], Dict[int, dict]]:

        df = pd.read_csv(path)

        #Tag名が"name"以外の場合は"name"に統一する
        if tag_name != "name":
            df["name"] = df.get(tag_name)

        df["category"] = df.get(category_name, "general")
        df["category"] = df["category"].fillna("general").astype(str).str.lower()

        if normalize:
            df["category"] = df.apply(
                lambda r: normalize(r["name"], r["category"]),
                axis=1
            )

        categories = set(df["category"])
        tags = df[["name", "category"]].to_dict(orient="index")

        return categories, tags
    
    #json読み込み。tagsは"name","category"に揃える   
    @staticmethod
    def load_json(
        path: str,
        schema: dict,
        normalize: Callable[[str, str], str] | None = None,
        encoding: str = "utf-8"
    ) -> Tuple[Set[str], Dict[int, dict]]:

        # --- JSON読み込み ---
        with open(path, encoding=encoding) as f:
            data = json.load(f)

        rows = []

        for item in schema["iterator"](data):
            row = {}

            # id（任意）
            if "id" in schema:
                row["id"] = schema["id"](item)

            # 出力は固定
            row["name"] = schema["name"](item)

            # category
            cat_fn = schema.get("category", lambda x: "general")

            if len(inspect.signature(cat_fn).parameters) == 2:
                row["category"] = cat_fn(item, data)
            else:
                row["category"] = cat_fn(item)

            rows.append(row)

        df = pd.DataFrame(rows)

        # category整形（CSVと統一）
        df["category"] = df.get("category", "general")
        df["category"] = df["category"].fillna("general").astype(str).str.lower()

        # normalize
        if normalize:
            df["category"] = df.apply(
                lambda r: normalize(r["name"], r["category"]),
                axis=1
            )

        # idソート
        if "id" in df.columns:
            df = df.sort_values("id").reset_index(drop=True)

        categories = set(df["category"])
        tags = df[["name", "category"]].to_dict(orient="index")

        return categories, tags    