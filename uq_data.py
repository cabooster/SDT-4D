"""Data loading and patch stitching helpers for UQ training and inference."""

import math
import random
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from data import list_tiffs


def _random_spatial_transform(volume):
    transform = random.randrange(8)
    if transform < 4:
        return np.rot90(volume, k=transform, axes=(-2, -1))
    return np.rot90(volume[..., ::-1], k=transform - 4, axes=(-2, -1))


def _training_time_gap(shape, patch_t, patch_x, gap_x, requested_size, stack_count):
    width_count = math.floor((shape[2] - patch_x) / gap_x) + 1
    height_count = math.floor((shape[1] - patch_x) / gap_x) + 1
    sample_count = math.ceil(requested_size / width_count / height_count / stack_count)
    sample_count = max(sample_count, 2)
    return max(math.floor((shape[0] - patch_t) / (sample_count - 1)), 1)


class UQTrainingPatchDataset(Dataset):
    """Reproduce the staged UQ script's multi-Z training patch extraction."""

    def __init__(
        self,
        data_dir,
        patch_z,
        patch_t,
        patch_x,
        overlap_factor,
        requested_size,
        max_frames,
        scale_factor,
    ):
        self.files = list_tiffs(data_dir)
        self.patch_z = patch_z
        self.patch_t = patch_t
        self.patch_x = patch_x
        self.volumes = []
        shape = None

        print("\033[1;31mImage list for training -----> \033[0m")
        print("All files are in -----> ", Path(data_dir).expanduser().resolve())
        print("Total stack number -----> ", len(self.files))
        print("Reading files...")
        print(
            "\033[1;33mPlease check the shape of these image stacks, since some "
            "hyperstacks have unusual shapes. In that case, re-save them with ImageJ. \033[0m"
        )
        for path in self.files:
            image = tifffile.imread(path)
            if image.ndim != 3:
                raise ValueError(f"Expected a (T,H,W) TIFF stack, got {image.shape}: {path}")
            if max_frames is not None:
                image = image[:max_frames]
            image = image.astype(np.float32) / float(scale_factor)
            print(path.name, " -----> the shape is", image.shape)
            if shape is None:
                shape = image.shape
            elif image.shape != shape:
                raise ValueError("All training TIFF stacks must have the same (T,H,W) shape")
            self.volumes.append(image)

        half_z = patch_z // 2
        if len(self.files) < patch_z:
            raise ValueError(f"Need at least {patch_z} Z slices for training")
        gap_x = int(patch_x * (1 - overlap_factor))
        if gap_x < 1:
            raise ValueError("The spatial patch gap must be at least one pixel")
        gap_t = _training_time_gap(
            shape, patch_t, patch_x, gap_x, requested_size, len(self.files)
        )
        if any(size < patch for size, patch in zip(shape, (patch_t, patch_x, patch_x))):
            raise ValueError(
                f"Image shape {shape} is smaller than patch shape "
                f"({patch_t}, {patch_x}, {patch_x})"
            )

        self.coordinates = []
        x_count = int((shape[1] - patch_x + gap_x) / gap_x)
        y_count = int((shape[2] - patch_x + gap_x) / gap_x)
        t_count = int((shape[0] - patch_t + gap_t) / gap_t)
        for center_z in range(half_z, len(self.files) - half_z):
            for x_index in range(x_count):
                for y_index in range(y_count):
                    for t_index in range(t_count):
                        self.coordinates.append(
                            (
                                center_z,
                                gap_t * t_index,
                                gap_x * x_index,
                                gap_x * y_index,
                            )
                        )
        if not self.coordinates:
            raise ValueError("No training patches could be generated")

    def __getitem__(self, index):
        center_z, start_t, start_h, start_w = self.coordinates[index]
        half_z = self.patch_z // 2
        patch = np.stack(
            [
                self.volumes[z][
                    start_t : start_t + self.patch_t,
                    start_h : start_h + self.patch_x,
                    start_w : start_w + self.patch_x,
                ]
                for z in range(center_z - half_z, center_z + half_z + 1)
            ]
        )
        patch = _random_spatial_transform(patch)
        return torch.from_numpy(np.expand_dims(patch.copy(), 0)).float()

    def __len__(self):
        return len(self.coordinates)


def _axis_coordinates(length, patch, gap):
    if length < patch:
        raise ValueError(f"Image dimension {length} is smaller than patch size {patch}")
    count = math.ceil((length - patch + gap) / gap)
    cut = (patch - gap) / 2
    result = []
    for index in range(count):
        start = gap * index if index != count - 1 else length - patch
        if index == 0:
            stack_start, stack_end = 0, int(patch - cut)
            patch_start, patch_end = 0, int(patch - cut)
        elif index == count - 1:
            stack_start, stack_end = int(length - patch + cut), length
            patch_start, patch_end = int(cut), patch
        else:
            stack_start = int(index * gap + cut)
            stack_end = int(index * gap + patch - cut)
            patch_start, patch_end = int(cut), int(patch - cut)
        result.append((start, stack_start, stack_end, patch_start, patch_end))
    return result


def make_uq_inference_coordinates(shape, patch_t, patch_x, gap_t, gap_x):
    """Create coordinates with the original UQ script's overlap cropping."""

    time_coords = _axis_coordinates(shape[0], patch_t, gap_t)
    height_coords = _axis_coordinates(shape[1], patch_x, gap_x)
    width_coords = _axis_coordinates(shape[2], patch_x, gap_x)
    coordinates = []
    for t_values in time_coords:
        for h_values in height_coords:
            for w_values in width_coords:
                t, st0, st1, pt0, pt1 = t_values
                h, sh0, sh1, ph0, ph1 = h_values
                w, sw0, sw1, pw0, pw1 = w_values
                coordinates.append(
                    {
                        "init_s": t,
                        "end_s": t + patch_t,
                        "init_h": h,
                        "end_h": h + patch_x,
                        "init_w": w,
                        "end_w": w + patch_x,
                        "stack_start_s": st0,
                        "stack_end_s": st1,
                        "patch_start_s": pt0,
                        "patch_end_s": pt1,
                        "stack_start_h": sh0,
                        "stack_end_h": sh1,
                        "patch_start_h": ph0,
                        "patch_end_h": ph1,
                        "stack_start_w": sw0,
                        "stack_end_w": sw1,
                        "patch_start_w": pw0,
                        "patch_end_w": pw1,
                    }
                )
    return coordinates


def _resolve_z_index(index, length, edge_mode):
    if 0 <= index < length:
        return index
    if edge_mode == "replicate":
        return min(max(index, 0), length - 1)
    if edge_mode == "reflect":
        if length == 1:
            return 0
        while index < 0 or index >= length:
            index = -index if index < 0 else 2 * length - 2 - index
        return index
    return None


def load_uq_z_window(files, center_z, patch_z, edge_mode, max_frames, scale_factor):
    """Load a Z window with configurable edge handling and no mean subtraction."""

    half_z = patch_z // 2
    indices = [
        _resolve_z_index(center_z + offset, len(files), edge_mode)
        for offset in range(-half_z, half_z + 1)
    ]
    if any(index is None for index in indices):
        return None
    volumes = []
    source_dtype = None
    shape = None
    for local_z, index in enumerate(indices):
        image = tifffile.imread(files[index])
        if image.ndim != 3:
            raise ValueError(f"Expected a (T,H,W) TIFF stack, got {image.shape}: {files[index]}")
        if max_frames is not None:
            image = image[:max_frames]
        if shape is None:
            shape = image.shape
        elif image.shape != shape:
            raise ValueError("All neighboring Z TIFF stacks must have the same (T,H,W) shape")
        if local_z == half_z:
            source_dtype = image.dtype
        print(files[index], " -----> the shape is", image.shape)
        volumes.append(image.astype(np.float32) / float(scale_factor))
    return volumes, 0.0, source_dtype


class UQInferencePatchDataset(Dataset):
    def __init__(self, volumes, coordinates):
        self.volumes = volumes
        self.coordinates = coordinates

    def __getitem__(self, index):
        coordinate = self.coordinates[index]
        patch = np.stack(
            [
                volume[
                    coordinate["init_s"] : coordinate["end_s"],
                    coordinate["init_h"] : coordinate["end_h"],
                    coordinate["init_w"] : coordinate["end_w"],
                ]
                for volume in self.volumes
            ]
        )
        return torch.from_numpy(np.expand_dims(patch, 0)).float(), coordinate

    def __len__(self):
        return len(self.coordinates)


def stitch_batch(output, coordinates, destination):
    """Copy the non-overlapping center of each output patch into a volume."""

    for index in range(output.shape[0]):
        values = {}
        for key, batch_values in coordinates.items():
            values[key] = int(batch_values[index].item())
        destination[
            values["stack_start_s"] : values["stack_end_s"],
            values["stack_start_h"] : values["stack_end_h"],
            values["stack_start_w"] : values["stack_end_w"],
        ] = output[
            index,
            values["patch_start_s"] : values["patch_end_s"],
            values["patch_start_h"] : values["patch_end_h"],
            values["patch_start_w"] : values["patch_end_w"],
        ]
