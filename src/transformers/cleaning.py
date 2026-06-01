import polars as pl
from ..base import BaseTransformer

class TextCleaningTransformer(BaseTransformer):
    """
    Performs basic text cleaning and normalization for parallel corpora.

    Params:
    - lowercase (bool): Convert all text to lowercase. Default: True.
    - normalize_whitespace (bool): Replace multiple spaces/tabs with a
single space. Default: True.
    """
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        cols = ["Source", "Target"]
        do_lowercase = self.params.get("lowercase", True)
        do_whitespace = self.params.get("normalize_whitespace", True)

        for col in cols:
            expr = pl.col(col)

            if do_whitespace:
                # Regex \s+ replaces all whitespace characters (tabs,

                expr = expr.str.replace_all(r"\s+", " ").str.strip_chars()

            if do_lowercase:
                expr = expr.str.to_lowercase()

            df = df.with_columns(expr.alias(col))

        return df