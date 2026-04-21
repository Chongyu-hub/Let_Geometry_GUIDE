"""
Qwen3-VL Demo: 3D Visual Grounding & 3D Video Object Detection Visualization

Usage:
    python demo.py --pretrained <checkpoint_path> [--task 3d_visual_grounding|3d_video_object_detection] \
                   [--device cuda:0] [--sample_idx 0] [--save_dir outputs/demo]

Example:
    python demo.py \
        --pretrained ckpt/qwen3_4b_vggt_sr_1e-5_0405_vggt_add_deepstack_7_10_13_16_19_22_step4_global_impgate_all \
        --task 3d_visual_grounding \
        --sample_idx 0
"""

import argparse
import copy
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as TF
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.image_utils import SizeDict

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Please install qwen-vl-utils: pip install qwen-vl-utils")

qwen_vl_finetune = Path(__file__).resolve().parent / "qwen-vl-finetune"
if qwen_vl_finetune.exists():
    sys.path.insert(0, str(qwen_vl_finetune))

from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwenvl.data.utils import load_and_preprocess_images


# ──────────────────────────────────────────────────────
#  Hyperparameters
# ──────────────────────────────────────────────────────
MIN_PIXELS: int = 256 * 28 * 28
MAX_PIXELS: int = 268324
MAX_NUM_FRAMES: int = 32
DEMO_DATA_DIR = "data/demo_data"

# VGGT patch size for geometry encoder
VGGT_PATCH_SIZE = 14

# A fixed set of distinct colors (BGR) for multi-object visualization
PALETTE = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (128, 0, 255),
    (255, 0, 128),
    (0, 255, 128),
]


# ──────────────────────────────────────────────────────
#  3D Bounding Box Drawing Utilities
# ──────────────────────────────────────────────────────
def get_obb_corners(center, extent, rotation_matrix):
    """Get 8 corners of an oriented bounding box."""
    half = extent / 2.0
    corners_local = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ]) * half  # (8, 3)
    corners_world = (rotation_matrix @ corners_local.T).T + center
    return corners_world


def project_points(pts_3d, extrinsic, intrinsic):
    """Project 3D points to 2D using extrinsic and intrinsic matrices."""
    ones = np.ones((pts_3d.shape[0], 1))
    pts_h = np.hstack([pts_3d, ones])  # (N, 4)
    pts_cam = (extrinsic @ pts_h.T).T  # (N, 4)
    pts_cam = pts_cam[:, :3]
    valid = pts_cam[:, 2] > 0.01
    pts_2d = (intrinsic[:3, :3] @ pts_cam.T).T
    pts_2d = pts_2d[:, :2] / (pts_2d[:, 2:3] + 1e-8)
    return pts_2d, valid


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
]


def draw_box3d_on_image(img, center, extent, rotation_matrix, extrinsic, intrinsic,
                        color=(0, 255, 0), label=None, thickness=2):
    """Draw a 3D bounding box projected onto an image."""
    corners = get_obb_corners(center, extent, rotation_matrix)
    pts_2d, valid = project_points(corners, extrinsic, intrinsic)

    if not np.any(valid):
        return img

    h, w = img.shape[:2]
    img_draw = img.copy()

    for i, j in EDGES:
        if valid[i] and valid[j]:
            p1 = (int(np.clip(pts_2d[i, 0], 0, w - 1)), int(np.clip(pts_2d[i, 1], 0, h - 1)))
            p2 = (int(np.clip(pts_2d[j, 0], 0, w - 1)), int(np.clip(pts_2d[j, 1], 0, h - 1)))
            cv2.line(img_draw, p1, p2, color, thickness, cv2.LINE_AA)

    if label is not None:
        vis_pts = pts_2d[valid]
        if len(vis_pts) > 0:
            min_y_idx = vis_pts[:, 1].argmin()
            tx = int(np.clip(vis_pts[min_y_idx, 0], 0, w - 10))
            ty = int(np.clip(vis_pts[min_y_idx, 1] - 8, 10, h - 1))
            font_scale = 0.5
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img_draw, (tx, ty - th - 4), (tx + tw + 4, ty + 2), color, -1)
            cv2.putText(img_draw, label, (tx + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return img_draw


# ──────────────────────────────────────────────────────
#  Inference Wrapper
# ──────────────────────────────────────────────────────
class Qwen3VLInference:
    def __init__(self, pretrained: str, device: str = "cuda:0"):
        self.device = device
        config = AutoConfig.from_pretrained(pretrained)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        self.processor = AutoProcessor.from_pretrained(
            pretrained, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS, padding_side="left"
        )

        self.use_geometry_encoder = getattr(config, "use_geometry_encoder", False)
        self.use_patch_size_alin = getattr(config, "use_patch_size_alin", False)

    @torch.no_grad()
    def call_model(
        self,
        contexts: list[str],
        visuals: list[list[Image.Image]],
        add_frame_index: bool = False,
        gen_kwargs: dict | None = None,
    ) -> list[str]:
        if gen_kwargs is None:
            gen_kwargs = {}

        batched_messages = []
        for i, context in enumerate(contexts):
            message = [{"role": "system", "content": "You are a helpful assistant."}]
            visual = visuals[i] if i < len(visuals) else None

            if visual is not None and isinstance(visual, (list, tuple)):
                image_content = []
                for idx, v in enumerate(visual):
                    if add_frame_index:
                        image_content.append({"type": "text", "text": f"Frame-{idx}: "})
                    image_content.append({
                        "type": "image", "image": v,
                        "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS,
                    })
                message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
            elif visual is not None and isinstance(visual, Image.Image):
                message.append({"role": "user", "content": [
                    {"type": "image", "image": visual, "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS},
                    {"type": "text", "text": context},
                ]})
            else:
                message.append({"role": "user", "content": [{"type": "text", "text": context}]})

            batched_messages.append(message)

        # Prepare geometry_encoder_inputs if enabled
        geometry_encoder_inputs = None
        if self.use_geometry_encoder:
            geometry_encoder_inputs = []
            to_tensor = TF.ToTensor()
            size_cfg = getattr(self.processor.image_processor, "size", {})
            min_pix = size_cfg.get("shortest_edge", 56 * 56)
            max_pix = size_cfg.get("longest_edge", 28 * 28 * 1280)
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

            for message in batched_messages:
                per_sample = []
                for msg in message:
                    if msg.get("role") != "user":
                        continue
                    for content in msg.get("content", []):
                        if not (isinstance(content, dict) and content.get("type") == "image"):
                            continue
                        img_obj = content.get("image")
                        if img_obj is None:
                            continue
                        pil_img = img_obj.convert("RGB") if isinstance(img_obj, Image.Image) else Image.open(img_obj).convert("RGB")
                        img_tensor = to_tensor(pil_img)
                        if not self.use_patch_size_alin:
                            _, height, width = img_tensor.shape
                            rh, rw = smart_resize(height, width, factor=VGGT_PATCH_SIZE,
                                                  min_pixels=min_pix, max_pixels=max_pix)
                            if (rh, rw) != (height, width):
                                img_tensor = self.processor.image_processor.resize(
                                    image=img_tensor,
                                    size=SizeDict(height=rh, width=rw),
                                    interpolation=self.processor.image_processor.resample,
                                )
                        per_sample.append(img_tensor)
                if per_sample:
                    shapes = [t.shape for t in per_sample]
                    if all(s == shapes[0] for s in shapes):
                        per_sample = torch.stack(per_sample)
                geometry_encoder_inputs.append(per_sample)

        texts = self.processor.apply_chat_template(
            batched_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(
            batched_messages, return_video_kwargs=False,
            image_patch_size=16, return_video_metadata=False,
        )
        inputs = self.processor(
            text=texts, images=image_inputs, videos=video_inputs,
            do_resize=False, return_tensors="pt",
        )

        # Align geometry_encoder_inputs to grid_thw if use_patch_size_alin
        if self.use_geometry_encoder and self.use_patch_size_alin and geometry_encoder_inputs is not None:
            grid_thw = inputs.get("image_grid_thw")
            if isinstance(grid_thw, torch.Tensor) and grid_thw.dim() == 2:
                grid_thw_list = grid_thw.tolist()
                grid_idx = 0
                resized_all = []
                for per_sample in geometry_encoder_inputs:
                    if isinstance(per_sample, torch.Tensor):
                        sample_list = [img for img in per_sample]
                    else:
                        sample_list = per_sample
                    resized_sample = []
                    for img in sample_list:
                        if not isinstance(img, torch.Tensor) or img.dim() != 3 or grid_idx >= len(grid_thw_list):
                            resized_sample.append(img)
                            grid_idx += 1
                            continue
                        _, grid_h, grid_w = grid_thw_list[grid_idx]
                        target_h = int(grid_h) * VGGT_PATCH_SIZE
                        target_w = int(grid_w) * VGGT_PATCH_SIZE
                        _, h, w = img.shape
                        if (target_h, target_w) != (h, w):
                            img = self.processor.image_processor.resize(
                                image=img,
                                size=SizeDict(height=target_h, width=target_w),
                                interpolation=self.processor.image_processor.resample,
                            )
                        resized_sample.append(img)
                        grid_idx += 1
                    if isinstance(per_sample, torch.Tensor) and resized_sample:
                        resized_sample = torch.stack(resized_sample)
                    resized_all.append(resized_sample)
                geometry_encoder_inputs = resized_all

        inputs = inputs.to(self.device)
        if self.use_geometry_encoder and geometry_encoder_inputs is not None:
            dev = inputs["input_ids"].device
            moved = []
            for ge in geometry_encoder_inputs:
                if isinstance(ge, torch.Tensor):
                    moved.append(ge.to(dev))
                elif isinstance(ge, list):
                    moved.append([g.to(dev) if isinstance(g, torch.Tensor) else g for g in ge])
                else:
                    moved.append(ge)
            inputs["geometry_encoder_inputs"] = moved

        gen_defaults = {
            "max_new_tokens": 4096,
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
        }
        gen_defaults.update(gen_kwargs)

        do_sample = gen_defaults["temperature"] is not None and gen_defaults["temperature"] > 0
        if not do_sample:
            gen_defaults["temperature"] = None
            gen_defaults["top_p"] = None

        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
            temperature=gen_defaults["temperature"],
            top_p=gen_defaults["top_p"],
            num_beams=gen_defaults["num_beams"],
            max_new_tokens=gen_defaults["max_new_tokens"],
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)
        ]
        answers = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return answers


# ──────────────────────────────────────────────────────
#  Visualization: 3D Visual Grounding
# ──────────────────────────────────────────────────────
def visualize_3d_visual_grounding(sample, prediction, media_dir, save_path=None, max_show=6):
    """Visualize prediction and GT bounding box for 3D visual grounding."""
    try:
        prediction = prediction.strip("\n").strip("```").strip("json").strip()
        bbox_list = [eval(prediction)]
    except Exception as e:
        print(f"[VG] Failed to parse prediction: {e}")
        return

    images = sample["images"]
    gt_bbox = sample["gt_bbox"]
    intrinsic = np.array(sample["cam2img"]).reshape(4, 4) if "cam2img" in sample else np.array([
        [1170.18798828125, 0, 647.75, 0],
        [0, 1170.18798828125, 483.75, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    show_indices = np.linspace(0, len(images) - 1, min(max_show, len(images)), dtype=int)
    np_images = []

    for i in show_indices:
        img_path = os.path.join(media_dir, images[i])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for item in bbox_list:
            if isinstance(item.get("bbox_3d"), str):
                item["bbox_3d"] = eval(item["bbox_3d"])
            frame_idx = int(item.get("frame", 0))
            center = np.array(item["bbox_3d"][:3])
            extent = np.array(item["bbox_3d"][3:6])
            rot = R.from_euler("ZXY", np.array(item["bbox_3d"][6:9])).as_matrix()
            extrinsic = np.linalg.inv(np.array(sample["cam2global"][frame_idx])) @ np.array(sample["cam2global"][i])
            img = draw_box3d_on_image(img, center, extent, rot, extrinsic, intrinsic,
                                      color=(0, 255, 0), label="pred", thickness=2)

        # Draw GT bbox (in axis-aligned world coordinates)
        gt_center = np.array(gt_bbox[:3])
        gt_extent = np.array(gt_bbox[3:6])
        gt_rot = np.eye(3)
        gt_extrinsic = np.array(sample["axis_align_matrix"]) @ np.array(sample["cam2global"][i])
        img = draw_box3d_on_image(img, gt_center, gt_extent, gt_rot, gt_extrinsic, intrinsic,
                                  color=(0, 0, 255), label="gt", thickness=2)
        np_images.append(img)

    ncols = min(3, len(np_images))
    nrows = (len(np_images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    for idx, ax in enumerate(axes_flat):
        if idx < len(np_images):
            ax.imshow(np_images[idx])
            ax.set_title(f"Frame {show_indices[idx]}", fontsize=10)
        ax.axis("off")
    fig.suptitle("3D Visual Grounding  (green=pred, blue=gt)", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────
#  Visualization: 3D Video Object Detection
# ──────────────────────────────────────────────────────
def visualize_3d_video_object_detection(sample, prediction, media_dir, save_path=None, max_show=6):
    """Visualize prediction bounding boxes for 3D video object detection."""
    try:
        prediction = prediction.strip("\n").strip("```").strip("json").strip()
        bbox_list = eval(prediction)
    except Exception as e:
        print(f"[Det] Failed to parse prediction: {e}")
        return

    label_to_color = {}
    color_idx = 0

    images = sample["images"]
    intrinsic = np.array(sample.get("cam2img", [
        [1170.18798828125, 0, 647.75, 0],
        [0, 1170.18798828125, 483.75, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])).reshape(4, 4)

    show_indices = np.linspace(0, len(images) - 1, min(max_show, len(images)), dtype=int)
    np_images = []

    for i in show_indices:
        img_path = os.path.join(media_dir, images[i])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for item in bbox_list:
            if isinstance(item.get("bbox_3d"), str):
                item["bbox_3d"] = eval(item["bbox_3d"])
            center = np.array(item["bbox_3d"][:3])
            extent = np.array(item["bbox_3d"][3:6])
            rot = R.from_euler("ZXY", np.array(item["bbox_3d"][6:9])).as_matrix()
            extrinsic = np.linalg.inv(np.array(sample["cam2global"][0])) @ np.array(sample["cam2global"][i])

            lbl = item.get("label", "object")
            if lbl not in label_to_color:
                label_to_color[lbl] = PALETTE[color_idx % len(PALETTE)]
                color_idx += 1

            img = draw_box3d_on_image(img, center, extent, rot, extrinsic, intrinsic,
                                      color=label_to_color[lbl], label=lbl, thickness=2)
        np_images.append(img)

    ncols = min(3, len(np_images))
    nrows = (len(np_images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()
    for idx, ax in enumerate(axes_flat):
        if idx < len(np_images):
            ax.imshow(np_images[idx])
            ax.set_title(f"Frame {show_indices[idx]}", fontsize=10)
        ax.axis("off")
    fig.suptitle("3D Video Object Detection", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────
#  Processing Pipelines
# ──────────────────────────────────────────────────────
def process_3d_visual_grounding(model, sample, media_dir, save_path=None):
    text = sample["prompt"]
    image_files = sample["images"]
    images = [
        Image.open(os.path.join(media_dir, f)).convert("RGB")
        for f in image_files
    ]

    output = model.call_model(
        contexts=[text],
        visuals=[images],
        add_frame_index=True,
    )[0]

    query_text = text.split("\n")[1] if "\n" in text else text
    print(f"Query : {query_text}")
    print(f"Output: {output}")

    visualize_3d_visual_grounding(sample, output, media_dir, save_path=save_path)
    return output


def process_3d_video_object_detection(model, sample, media_dir, save_path=None):
    text = sample["conversations"][0]["value"].replace("<image>", "")
    image_files = sample["images"]
    images = [
        Image.open(os.path.join(media_dir, f)).convert("RGB")
        for f in image_files
    ]

    output = model.call_model(
        contexts=[text],
        visuals=[images],
    )[0]

    print(f"Output: {output}")

    visualize_3d_video_object_detection(sample, output, media_dir, save_path=save_path)
    return output


# ──────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 3D Visualization Demo")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--task", type=str, default="3d_visual_grounding",
                        choices=["3d_visual_grounding", "3d_video_object_detection", "all"],
                        help="Task to run")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index in the demo dataset")
    parser.add_argument("--demo_data_dir", type=str, default=DEMO_DATA_DIR, help="Path to demo data")
    parser.add_argument("--save_dir", type=str, default="outputs/demo", help="Directory to save visualization")
    parser.add_argument("--max_show", type=int, default=6, help="Max number of frames to display")
    args = parser.parse_args()

    global MAX_PIXELS, MIN_PIXELS
    # Override device
    print(f"Loading model from {args.pretrained} on {args.device} ...")
    model = Qwen3VLInference(args.pretrained, device=args.device)
    print("Model loaded successfully.")

    tasks_to_run = (
        ["3d_visual_grounding", "3d_video_object_detection"] if args.task == "all" else [args.task]
    )

    for task in tasks_to_run:
        data_path = os.path.join(args.demo_data_dir, task, "samples.json")
        if not os.path.exists(data_path):
            print(f"[SKIP] Data not found: {data_path}")
            continue

        with open(data_path) as f:
            data = json.load(f)
        print(f"\n{'=' * 60}")
        print(f"Task: {task}  |  {len(data)} samples loaded  |  running sample {args.sample_idx}")
        print(f"{'=' * 60}")

        media_dir = os.path.join(args.demo_data_dir, task, "media")
        sample = data[args.sample_idx]
        save_path = os.path.join(args.save_dir, f"{task}_sample{args.sample_idx}.png")

        if task == "3d_visual_grounding":
            process_3d_visual_grounding(model, sample, media_dir, save_path=save_path)
        elif task == "3d_video_object_detection":
            process_3d_video_object_detection(model, sample, media_dir, save_path=save_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
