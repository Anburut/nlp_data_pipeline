import polars as pl
import re
from ..base import BaseTransformer

class OromoLinguisticTransformer(BaseTransformer):
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:

        # Phase 4: Orthography Normalization
        def normalize_qubee(text):
            if text is None:
                return None
            return re.sub(r"[\u02B0\u02BC`\u2019]", "'", text)

        # Phase 6: Basic NER/Numeric Consistency Check
        def check_numeric_consistency(row):
            src, tgt = row["Source"], row["Target"]
            if not src or not tgt:
                return "CONSISTENT"
            nums_src = re.findall(r'\d+', src)
            nums_tgt = re.findall(r'\d+', tgt)
            if nums_src and nums_tgt and sorted(nums_src) != sorted(nums_tgt):  # ← joined
                return "NUM_MISMATCH"
            return "CONSISTENT"

        df = df.with_columns([
            pl.col("Target").map_elements(normalize_qubee, return_dtype=pl.String),  # ← joined
            pl.struct(["Source", "Target"]).map_elements(
                check_numeric_consistency, return_dtype=pl.String  # ← joined
            ).alias("NER_Status")
        ])
        return df