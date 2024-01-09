from pydantic import BaseModel
from typing import List


class TrainConfig(BaseModel):
    classes: List[str]
    epochs: int
    batch_size: int