import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from octo.model.octo_model import OctoModel


def show(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}/{key}" if prefix else key
            show(item, name)
    else:
        try:
            array = np.asarray(value)
            print(
                f"{prefix}: shape={array.shape}, "
                f"dtype={array.dtype}"
            )
        except Exception:
            print(f"{prefix}: {type(value).__name__}")


print("Loading Octo-Small...")
model = OctoModel.load_pretrained(
    "hf://rail-berkeley/octo-small-1.5"
)

print("\nEXAMPLE BATCH STRUCTURE")
show(model.example_batch)
