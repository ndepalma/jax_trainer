from typing import Optional

import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from ml_collections import ConfigDict
from datasets import Dataset


@dataclass
class DatasetModule:
    """Data module class that holds the datasets and data loaders."""

    config: ConfigDict
    train: Optional[Dataset]
    val: Optional[Dataset]
    test: Optional[Dataset]
    # train_loader: Optional[Dataset]
    # val_loader: Optional[Dataset]
    # test_loader: Optional[Dataset]
    metadata: Optional[dict] = None


@dataclass
class Batch:
    """Base class for batches.

    Attribute `size` is required and used, e.g. for logging.
    """

    size: int
    # Add any additional batch information here

    def __getitem__(self, key):
        vals = {}
        if isinstance(key, int):
            vals["size"] = 1
        for k, v in self.__dict__.items():
            if k == "size":
                continue
            if isinstance(v, (np.ndarray, jnp.ndarray)):
                vals[k] = v[key]
                if "size" not in vals:
                    vals["size"] = vals[k].shape[0]
            else:
                vals[k] = v
        return self.__class__(**vals)


@dataclass
class SupervisedBatch(Batch):
    """Extension of the base batch class for supervised learning."""

    input: np.ndarray
    target: np.ndarray
