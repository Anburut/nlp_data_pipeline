from abc import ABC, abstractmethod
import polars as pl

class BaseTransformer(ABC):

    def __init__(self, params: dict = None):
        # Store configuration parameters for the transformer
        self.params = params or {}

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Processes the input Polars DataFrame and returns a modified version.
        :param df: The input dataset as a Polars DataFrame.
        :return: The processed Polars DataFrame.
        """
        pass