import polars as pl
import re
from ..base import BaseTransformer


class FilteringTransformer(BaseTransformer):
    """
    Implements token quality, deduplication, and length ratio filtering.
    """

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:

        # 1. Exact deduplication
        before = len(df)
        df = df.unique(subset=["Source", "Target"])
        print(f"    Deduplication: removed {before - len(df)}")

        # 2. Token quality — remove pairs under 3 tokens
        before = len(df)
        df = df.filter(
            (pl.col("Source").str.split(" ").list.len() >= 3) &
            (pl.col("Target").str.split(" ").list.len() >= 3)
        )
        print(f"    Min token filter: removed {before - len(df)}")

        # 3. Remove symbol-only or numeric-only targets
        before = len(df)
        df = df.filter(~pl.col("Target").str.contains(r'^[\d\s\W]+$'))
        print(f"    Symbol-only filter: removed {before - len(df)}")

        # 4. Length ratio — relaxed to 0.8–3.5 to retain short but valid pairs
        #    Your corpus shows 69.7% of pairs in 1.0–1.5 range, so 1.2 floor
        #    was cutting the majority. Oromo agglutination varies by sentence type.
        min_ratio = self.params.get("min_ratio", 0.8)
        max_ratio = self.params.get("max_ratio", 3.5)
        before = len(df)
        df = df.with_columns([
            (
                pl.col("Target").str.len_chars().cast(pl.Float64) /
                pl.col("Source").str.len_chars().cast(pl.Float64)
            ).alias("ratio")
        ])
        df = df.filter(pl.col("ratio").is_between(min_ratio, max_ratio))
        df = df.drop("ratio")
        print(f"    Ratio filter ({min_ratio}-{max_ratio}): removed {before - len(df)}")

        return df