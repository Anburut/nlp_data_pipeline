import polars as pl
from ..base import BaseTransformer

class TokenizationTransformer(BaseTransformer):
    """
    Modular tokenization stage.

    Params:
    - type (str): The tokenization strategy. Currently supports
"whitespace".
"""
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        tokenizer_type = self.params.get("type", "whitespace")

        def tokenize(text):
            if text is None:
                return []
            if tokenizer_type == "whitespace":
                return text.split()

            # Future expansion: Add HuggingFace AutoTokenizer here
            return text.split()

    # We create new columns for tokens to keep the original text intact
        df = df.with_columns([
            pl.col("Source").map_elements(tokenize,
return_dtype=pl.List(pl.String)).alias("tokens_src"),
    pl.col("Target").map_elements(tokenize,
return_dtype=pl.List(pl.String)).alias("tokens_tgt")
    ])

        return df
