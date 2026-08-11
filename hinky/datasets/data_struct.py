"""Data structures for datasets."""
from pathlib import Path
from typing import Annotated, Generic, TypeVar

from datasets.table import ConcatenationTable, InMemoryTable, MemoryMappedTable
from pyarrow import Table
from pydantic import BaseModel, ConfigDict, Field

ConfigType = TypeVar("ConfigType", bound=BaseModel)
PermissibleHFTables = ConcatenationTable | InMemoryTable | MemoryMappedTable
PermissibleArrowTables = Table
TableType = TypeVar("TableType", bound=PermissibleHFTables | PermissibleArrowTables)

class DatasetModule(BaseModel, Generic[ConfigType, TableType]):
  """Data module class that holds the datasets and data loaders."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  config: ConfigType
  train: TableType
  test: TableType
  val: TableType | None = None
  metadata: dict | None = None

class DatasetTransform(BaseModel): 
  columns_req: list[str] = []
  columns_added: list[str] = []

class NormalizeImageTransform(DatasetTransform):
  normalize_column: bool = False
  columns_req: list[str] = ["image"]
  columns_added: list[str] = ["n_image"]

class PadResizeImageTransform(DatasetTransform):
  pad_resize: bool = False
  desired_square_resolution: int = 200
  columns_req: list[str] = ["image"]
  columns_added: list[str] = ["params"]

class PrepareDatasetConfig(BaseModel):
  create_validation_set: bool = False
  image_transforms: list[NormalizeImageTransform | PadResizeImageTransform] = []

class HuggingFaceDatasetConfig(BaseModel):
  hf_dataset_uri: str

class CachedDatasetConfig(BaseModel):
  cache_path: Path

class TrainingDatasetConfig(BaseModel):
  limit_to: int | None = None
  batch_size: Annotated[int, Field(frozen=True, gt=1)]
  class_names: list[str] = Field(default_factory=list)

class FullDatasetSpecification(BaseModel):
  source: HuggingFaceDatasetConfig | CachedDatasetConfig
  preparation: PrepareDatasetConfig = Field(default_factory=PrepareDatasetConfig)
  training_params: TrainingDatasetConfig
