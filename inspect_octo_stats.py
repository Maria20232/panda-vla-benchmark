import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from octo.model.octo_model import OctoModel


def print_nested(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            print_nested(value, new_prefix)

    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}: {type(obj).__name__}, length={len(obj)}")

    else:
        try:
            arr = np.asarray(obj)

            if arr.ndim == 0:
                print(f"{prefix}: {arr.item()}")
            else:
                print(
                    f"{prefix}: shape={arr.shape}, "
                    f"dtype={arr.dtype}, values={arr}"
                )

        except Exception:
            print(f"{prefix}: {type(obj).__name__} = {obj}")


def main():
    print("Loading Octo-Small...")

    model = OctoModel.load_pretrained(
        "hf://rail-berkeley/octo-small-1.5"
    )

    print("\nModel dataset statistics:")
    print_nested(model.dataset_statistics)

    print("\nExample batch observation keys:")

    if hasattr(model, "example_batch"):
        print_nested(model.example_batch)
    else:
        print("No example_batch attribute found.")


if __name__ == "__main__":
    main()
