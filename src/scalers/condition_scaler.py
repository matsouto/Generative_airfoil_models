import numpy as np
import pandas as pd


class ConditionScaler:
    def __init__(self):
        self.condition_dim = None
        self.min_value = None
        self.max_value = None

    def fit(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(
                "Expected condition values with shape (N, condition_dim), got "
                f"{values.shape}."
            )

        self.condition_dim = values.shape[1]
        if self.condition_dim == 0:
            raise ValueError(
                "Expected at least one condition column to fit ConditionScaler."
            )

        self.min_value = np.min(values, axis=0)
        self.max_value = np.max(values, axis=0)
        identical_bounds = self.max_value == self.min_value
        self.max_value[identical_bounds] += 1e-6

    def transform(self, values):
        if self.min_value is None or self.max_value is None:
            raise RuntimeError("ConditionScaler must be fitted before transform().")

        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.condition_dim:
            raise ValueError(
                "Expected condition values with shape "
                f"(N, {self.condition_dim}), got {values.shape}."
            )

        return (
            2 * (values - self.min_value) / (self.max_value - self.min_value) - 1
        ).astype(np.float32)

    def to_metadata(self, condition_columns):
        if self.min_value is None or self.max_value is None:
            raise RuntimeError("ConditionScaler must be fitted before serialization.")

        return {
            "condition_columns": list(condition_columns),
            "min": self.min_value.astype(float).tolist(),
            "max": self.max_value.astype(float).tolist(),
        }
