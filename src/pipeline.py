import polars as pl
from typing import List, Dict, Any, Tuple
from .base import BaseTransformer
from .transformers.cleaning import TextCleaningTransformer
from .transformers.tokenization import TokenizationTransformer
from .transformers.filtering import FilteringTransformer
from .transformers.validation import ValidationTransformer
from .transformers.linguistic import OromoLinguisticTransformer


class NLPPipeline:
    def __init__(self, config: List[Dict[str, Any]]):
        self.config = config
        self._registry = {
            "cleaning": TextCleaningTransformer,
            "linguistic": OromoLinguisticTransformer,
            "tokenization": TokenizationTransformer,
            "filtering": FilteringTransformer,
            "validation": ValidationTransformer,
        }
        self.transformers = self._build_pipeline()

    def _build_pipeline(self) -> List[BaseTransformer]:
        return [
            self._registry[s["step"]](s.get("params", {}))
            for s in self.config
        ]

    def _flatten_nested_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        for col_name in df.columns:
            dtype = df.schema[col_name]
            if dtype == pl.List or (hasattr(dtype, 'is_nested') and dtype.is_nested()):
                df = df.with_columns(
                    pl.col(col_name).list.join(" ").alias(col_name)
                )
        return df

    def run_research_mode(
        self,
        df: pl.DataFrame,
        output_path: str,
        audit_path: str,
        review_path: str,
        format: str = "jsonl",
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:

        # 1. Separate kept rows from removed
        clean_df = df.filter(pl.col("Decision") != "REMOVE")
        removed_df = df.filter(pl.col("Decision") == "REMOVE")

        # 2. Export audit log
        removed_df.write_csv(audit_path)

        # 3. Run transformer chain
        for transformer in self.transformers:
            clean_df = transformer.transform(clean_df)

        # 4. Ratio statistics — FIXED: Target/Source (OM/EN), not inverted
        ratio_df = clean_df.select([
            (
                pl.col("Target").str.len_chars().cast(pl.Float64)
                / pl.col("Source").str.len_chars().cast(pl.Float64)
            ).alias("ratio")
        ])

        # 5. Flatten nested columns for CSV/TSV
        if format in ["csv", "tsv"]:
            clean_df = self._flatten_nested_columns(clean_df)

        # 6. Export
        if format == "jsonl":
            clean_df.write_ndjson(output_path)
        elif format == "csv":
            clean_df.write_csv(output_path)
        elif format == "tsv":
            clean_df.write_csv(output_path, separator='\t')
        else:
            raise ValueError(f"Unsupported format: {format}")

        return clean_df, ratio_df