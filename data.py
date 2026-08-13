"""TIFF loading and patch extraction used by training and inference."""

from pathlib import Path
import math
import random

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


TIFF_SUFFIXES = {".tif", ".tiff"}


def list_tiffs(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"TIFF directory does not exist: {folder}")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files found in: {folder}")
    return files


def validate_patch_args(patch_z, patch_t, patch_x, overlap_factor, scale_factor=1.0):
    if patch_z < 1 or patch_z % 2 == 0:
        raise ValueError("--patch-z must be a positive odd number")
    if patch_t < 16 or patch_t % 16:
        raise ValueError("--patch-t must be a positive multiple of 16")
    if patch_x < 2 or patch_x % 2:
        raise ValueError("--patch-x must be a positive even number")
    if not 0 <= overlap_factor < 1:
        raise ValueError("--overlap-factor must be in [0, 1)")
    if scale_factor <= 0:
        raise ValueError("--scale-factor must be greater than zero")


def _read_tiff(path, max_frames=None, scale_factor=1.0):
    image = tifffile.imread(path)
    if image.ndim != 3:
        raise ValueError(f"Expected a (T,H,W) TIFF stack, got {image.shape}: {path}")
    if max_frames is not None:
        image = image[:max_frames]
    dtype = image.dtype
    image = image.astype(np.float32) / float(scale_factor)
    return image, dtype


def _axis_starts(length, patch, gap):
    if length < patch:
        raise ValueError(f"Image dimension {length} is smaller than patch size {patch}")
    if length == patch:
        return [0]
    count = math.ceil((length - patch) / gap) + 1
    return sorted(set(min(i * gap, length - patch) for i in range(count)))


def _axis_crops(starts, patch, length):
    """Return stack and local patch ranges with midpoint overlap cropping."""
    ranges = []
    for i, start in enumerate(starts):
        stack_start = 0 if i == 0 else (starts[i - 1] + patch + start) // 2
        stack_end = length if i == len(starts) - 1 else (start + patch + starts[i + 1]) // 2
        ranges.append((stack_start, stack_end, stack_start - start, stack_end - start))
    return ranges


def make_inference_coordinates(shape, patch_t, patch_x, gap_t, gap_x):
    time, height, width = shape
    t_starts = _axis_starts(time, patch_t, gap_t)
    h_starts = _axis_starts(height, patch_x, gap_x)
    w_starts = _axis_starts(width, patch_x, gap_x)
    t_crops = _axis_crops(t_starts, patch_t, time)
    h_crops = _axis_crops(h_starts, patch_x, height)
    w_crops = _axis_crops(w_starts, patch_x, width)

    coordinates = []
    for ti, init_t in enumerate(t_starts):
        for hi, init_h in enumerate(h_starts):
            for wi, init_w in enumerate(w_starts):
                st0, st1, pt0, pt1 = t_crops[ti]
                sh0, sh1, ph0, ph1 = h_crops[hi]
                sw0, sw1, pw0, pw1 = w_crops[wi]
                coordinates.append(
                    {
                        "init_s": init_t,
                        "end_s": init_t + patch_t,
                        "init_h": init_h,
                        "end_h": init_h + patch_x,
                        "init_w": init_w,
                        "end_w": init_w + patch_x,
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


def load_z_window(files, center_z, patch_z, edge_mode, max_frames, scale_factor):
    half_z = patch_z // 2
    indices = [
        _resolve_z_index(center_z + offset, len(files), edge_mode)
        for offset in range(-half_z, half_z + 1)
    ]
    if any(index is None for index in indices):
        return None

    volumes = []
    center_mean = None
    center_dtype = None
    expected_shape = None
    for local_z, index in enumerate(indices):
        volume, dtype = _read_tiff(files[index], max_frames, scale_factor)
        if expected_shape is None:
            expected_shape = volume.shape
        elif volume.shape != expected_shape:
            raise ValueError("All neighboring Z TIFF stacks must have the same (T,H,W) shape")
        mean = float(volume.mean())
        volumes.append(volume - mean)
        if local_z == half_z:
            center_mean, center_dtype = mean, dtype
    return volumes, center_mean, center_dtype


def _random_spatial_transform(volume):
    transform = random.randrange(8)
    if transform < 4:
        return np.rot90(volume, k=transform, axes=(-2, -1))
    return np.rot90(volume[..., ::-1], k=transform - 4, axes=(-2, -1))


class TrainingPatchDataset(Dataset):
    def __init__(
        self,
        files,
        patch_z,
        patch_t,
        patch_x,
        overlap_factor,
        requested_size,
        max_frames,
        scale_factor,
        seed=0,
    ):
        self.patch_z = patch_z
        self.volumes = []
        shape = None
        for path in files:
            volume, _ = _read_tiff(path, max_frames, scale_factor)
            volume -= volume.mean()
            if shape is None:
                shape = volume.shape
            elif volume.shape != shape:
                raise ValueError("All training TIFF stacks must have the same (T,H,W) shape")
            self.volumes.append(volume)

        gap_t = max(1, int(patch_t * (1 - overlap_factor)))
        gap_x = max(1, int(patch_x * (1 - overlap_factor)))
        t_starts = _axis_starts(shape[0], patch_t, gap_t)
        h_starts = _axis_starts(shape[1], patch_x, gap_x)
        w_starts = _axis_starts(shape[2], patch_x, gap_x)
        half_z = patch_z // 2
        coordinates = [
            (z, t, h, w)
            for z in range(half_z, len(files) - half_z)
            for t in t_starts
            for h in h_starts
            for w in w_starts
        ]
        if not coordinates:
            raise ValueError(f"Need at least {patch_z} Z slices for training")

        self.all_coordinates = coordinates
        self.requested_size = requested_size
        self.seed = seed
        self.resample(epoch=0)
        self.patch_t = patch_t
        self.patch_x = patch_x

    def resample(self, epoch):
        """Select a deterministic, fresh coordinate subset for an epoch."""
        rng = random.Random(self.seed + epoch)
        if self.requested_size <= 0:
            self.coordinates = list(self.all_coordinates)
        elif self.requested_size <= len(self.all_coordinates):
            self.coordinates = rng.sample(self.all_coordinates, self.requested_size)
        else:
            self.coordinates = [rng.choice(self.all_coordinates) for _ in range(self.requested_size)]

    def __getitem__(self, index):
        center_z, t, h, w = self.coordinates[index]
        half_z = self.patch_z // 2
        patch = np.stack(
            [
                self.volumes[z][t : t + self.patch_t, h : h + self.patch_x, w : w + self.patch_x]
                for z in range(center_z - half_z, center_z + half_z + 1)
            ]
        )
        patch = _random_spatial_transform(patch)
        return torch.from_numpy(np.expand_dims(patch.copy(), 0)).float()

    def __len__(self):
        return len(self.coordinates)


class InferencePatchDataset(Dataset):
    def __init__(self, volumes, coordinates):
        self.volumes = volumes
        self.coordinates = coordinates

    def __getitem__(self, index):
        coord = self.coordinates[index]
        patch = np.stack(
            [
                volume[
                    coord["init_s"] : coord["end_s"],
                    coord["init_h"] : coord["end_h"],
                    coord["init_w"] : coord["end_w"],
                ]
                for volume in self.volumes
            ]
        )
        return torch.from_numpy(np.expand_dims(patch, 0)).float(), coord

    def __len__(self):
        return len(self.coordinates)
