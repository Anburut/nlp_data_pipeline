import polars as pl
from ..base import BaseTransformer

class ValidationTransformer(BaseTransformer):
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        # If the Analyzer provided a Status, we strictly keep CLEAN rows
        if "Status" in df.columns:
            df = df.filter(pl.col("Status") == "CLEAN")
        # Safety check for nulls
        df = df.filter(
            pl.col("Source").is_not_null() &
            (pl.col("Source").str.len_chars() > 0) &  # ← joined
            pl.col("Target").is_not_null() &
            (pl.col("Target").str.len_chars() > 0)    # ← joined
        )
        return df