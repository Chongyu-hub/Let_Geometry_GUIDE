import argparse

import torch

from qwenvl.model.vggt.models.vggt import VGGT
from qwenvl.model.vggt.utils.load_fn import load_and_preprocess_images


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    debug_image_paths = [
        "/storage/chongyu/Qwen3-VL/data/media/scannet/posed_images/scene0000_00/00000.jpg",
        # "/path/to/imageB.png",
        # "/path/to/imageC.png",
    ]
    model = VGGT.from_pretrained("/storage/chongyu/Qwen3-VL/ckpt/VGGT-1B").to(device)
    images = load_and_preprocess_images(debug_image_paths).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, patch_start_idx = model.aggregator(images[None])
            # predictions = model(images)

    print("VGGT run OK.")
    # for key, value in predictions.items():
    #     if torch.is_tensor(value):
    #         print(f"{key}: {tuple(value.shape)} {value.dtype}")
    #     else:
    #         print(f"{key}: {type(value)}")


if __name__ == "__main__":
    main()

