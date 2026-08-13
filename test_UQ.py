"""Run SDT-4D UQ inference and save uncertainty products."""

import argparse
import datetime
import math
import os
from pathlib import Path
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from common import load_weights, maybe_data_parallel, prepare_device, str2bool
from data import list_tiffs
from SDT4D import SDT4D
from uq_data import (
    UQInferencePatchDataset,
    load_uq_z_window,
    make_uq_inference_coordinates,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--GPU",
        type=str,
        default="4,5,6,7",
        help="GPU indices used for computation, for example '0' or '0,1,2'",
    )
    parser.add_argument(
        "--denoise_model",
        type=str,
        default="noise_2RPN_alpha_beta_202608111739_bayesian_mu2nll_smooth",
        help="Folder containing checkpoints to test",
    )
    parser.add_argument(
        "--datasets_folder",
        type=str,
        default="/data/liyixin/01_noisy_datasets/noise_2RPN_alpha_beta",
        help="Folder containing input TIFF stacks",
    )
    parser.add_argument("--patch_x", type=int, default=64, help="Patch size in X and Y")
    parser.add_argument("--patch_t", type=int, default=64, help="Patch size in T")
    parser.add_argument("--patch_z", type=int, default=5, help="Patch size in Z")
    parser.add_argument("--overlap_factor", type=float, default=0.5)
    parser.add_argument(
        "--z_edge_mode",
        choices=("skip", "replicate", "reflect"),
        default="replicate",
        help="Z-boundary handling for edge TIFF stacks",
    )
    parser.add_argument("--datasets_path", type=str, default="", help="Dataset root path")
    parser.add_argument("--pth_path", type=str, default="./pth", help="Checkpoint root path")
    parser.add_argument("--output_path", type=str, default="./results", help="Output directory")
    parser.add_argument("--test_datasize", type=int, default=1000000)
    parser.add_argument("--scale_factor", type=float, default=1)
    parser.add_argument("--bayesian", type=str2bool, default=True)
    parser.add_argument("--mc_samples", type=int, default=6)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=20,
        help="Half-interval width for Gaussian confidence computation",
    )
    parser.add_argument("--reliability_bins", type=int, default=50)
    parser.add_argument(
        "--save_dtype",
        choices=("float32", "float16", "uint16", "uint8"),
        default="float32",
        help="Data type for UQ maps",
    )
    parser.add_argument(
        "--save_scale_mode",
        choices=("none", "max", "p99", "p99_9"),
        default="p99",
        help="Scaling mode for integer UQ maps",
    )
    parser.add_argument(
        "--tiff_compress",
        choices=("none", "zlib", "lzma", "zstd", "lzw"),
        default="none",
        help="TIFF compression for UQ maps",
    )
    parser.add_argument(
        "--checkpoint_slice",
        default="latest",
        help="Use 'latest' or a Python-style start:stop:step checkpoint selection",
    )
    parser.add_argument(
        "--selected_z_test",
        default="all",
        help="Comma-separated zero-based Z centers, or 'all'",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--gt_dir",
        default="/data/liyixin/01_noisy_datasets/clean-3D-ab",
        help="Ground-truth TIFF directory; use an empty value to disable evaluation",
    )
    parser.add_argument(
        "--gt_pattern",
        default="clean_depth{depth}um_scale0.80_1500frames_preprocessed.tif",
        help="Optional ground-truth filename template using {name}, {stem}, or {depth}",
    )
    return parser.parse_args()


def build_uq_model(patch_x, patch_t, bayesian):
    return SDT4D(
        img_dim=patch_x,
        img_time=patch_t,
        in_channel=1,
        embedding_dim=128,
        num_heads=8,
        hidden_dim=128 * 4,
        window_size=7,
        num_transBlock=1,
        attn_dropout_rate=0.1,
        f_maps=[8, 16, 32, 64],
        input_dropout_rate=0,
        bayesian=bayesian,
    )


def save_test_config(opt, path):
    """Write the same test fields as the original UQ entry point."""

    fields = (
        "datasets_path",
        "datasets_folder",
        "denoise_model",
        "pth_path",
        "output_path",
        "GPU",
        "batch_size",
        "patch_x",
        "patch_y",
        "patch_t",
        "gap_x",
        "gap_y",
        "gap_t",
        "test_datasize",
        "scale_factor",
        "epsilon",
    )
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump({field: getattr(opt, field) for field in fields}, file, sort_keys=False)


def parse_slice(specification):
    parts = specification.split(":")
    if len(parts) != 3:
        raise ValueError("--checkpoint_slice must use start:stop:step syntax")
    values = [int(value) if value else None for value in parts]
    if values[2] == 0:
        raise ValueError("--checkpoint_slice step cannot be zero")
    return slice(*values)


def parse_z_indices(specification, count):
    if specification.strip().lower() == "all":
        return list(range(count))
    indices = sorted({int(value.strip()) for value in specification.split(",") if value.strip()})
    invalid = [index for index in indices if index < 0 or index >= count]
    if invalid:
        raise ValueError(f"Selected Z indices outside [0, {count - 1}]: {invalid}")
    return indices


def gaussian_interval_confidence(mean_images, sigma_images, epsilon):
    mean_stack = np.stack(mean_images, axis=0)
    sigma_stack = np.maximum(np.stack(sigma_images, axis=0), 1e-12)
    mean_average = np.mean(mean_stack, axis=0)
    upper = (mean_average + epsilon - mean_stack) / (np.sqrt(2.0) * sigma_stack)
    lower = (mean_average - epsilon - mean_stack) / (np.sqrt(2.0) * sigma_stack)
    return np.mean(0.5 * (erf(upper) - erf(lower)), axis=0).astype(np.float32)


def softplus_np(value):
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0)


def enable_mc_dropout(model, enabled):
    if not enabled:
        return
    base = model.module if isinstance(model, nn.DataParallel) else model
    for layer in base.modules():
        if isinstance(layer, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            layer.train()


def save_tif_scaled(array, path, save_dtype, scale_mode, compression, scale_factor=1.0):
    array = array.squeeze().astype(np.float32) * float(scale_factor)
    compression_value = None if compression == "none" else compression
    if save_dtype == "float32":
        tifffile.imwrite(path, array.astype(np.float32), compression=compression_value)
        return
    if save_dtype == "float16":
        tifffile.imwrite(path, array.astype(np.float16), compression=compression_value)
        return
    integer_max = 65535.0 if save_dtype == "uint16" else 255.0
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        maximum = 1.0
    elif scale_mode == "p99":
        maximum = float(np.percentile(finite, 99.0))
    elif scale_mode == "p99_9":
        maximum = float(np.percentile(finite, 99.9))
    else:
        maximum = float(np.max(finite))
    if not np.isfinite(maximum) or maximum <= 0:
        maximum = 1.0
    scaled = np.round(np.clip(array, 0.0, maximum) / maximum * integer_max)
    dtype = np.uint16 if save_dtype == "uint16" else np.uint8
    tifffile.imwrite(
        path,
        scaled.astype(dtype),
        compression=compression_value,
        description=f"scaled_max={maximum:.6g};dtype={save_dtype}",
    )


def coordinate_value(coordinates, key, index):
    value = coordinates[key][index]
    return int(value.item()) if torch.is_tensor(value) else int(value)


def stitch_scaled_patch(mean_patch, raw_patch, coordinates, index, destination):
    keys = (
        "stack_start_w",
        "stack_end_w",
        "patch_start_w",
        "patch_end_w",
        "stack_start_h",
        "stack_end_h",
        "patch_start_h",
        "patch_end_h",
        "stack_start_s",
        "stack_end_s",
        "patch_start_s",
        "patch_end_s",
    )
    values = {key: coordinate_value(coordinates, key, index) for key in keys}
    output_crop = mean_patch[
        index,
        values["patch_start_s"] : values["patch_end_s"],
        values["patch_start_h"] : values["patch_end_h"],
        values["patch_start_w"] : values["patch_end_w"],
    ]
    raw_crop = raw_patch[
        index,
        values["patch_start_s"] : values["patch_end_s"],
        values["patch_start_h"] : values["patch_end_h"],
        values["patch_start_w"] : values["patch_end_w"],
    ]
    scale = math.sqrt(max(0.0, float(np.sum(raw_crop))) / max(float(np.sum(output_crop)), 1e-12))
    slices = (
        slice(values["stack_start_s"], values["stack_end_s"]),
        slice(values["stack_start_h"], values["stack_end_h"]),
        slice(values["stack_start_w"], values["stack_end_w"]),
    )
    destination[slices] = output_crop * scale
    return scale, slices, values


def calculate_snr(estimate, ground_truth):
    noise = estimate - ground_truth
    return 20 * math.log10(
        math.sqrt(float(np.sum(ground_truth**2))) / math.sqrt(float(np.sum(noise**2)))
    )


def find_ground_truth(input_path, gt_dir, pattern):
    if not gt_dir:
        return None
    directory = Path(gt_dir).expanduser().resolve()
    depth_match = re.search(r"depth(\d+)um", input_path.name)
    depth = depth_match.group(1) if depth_match else ""
    if pattern:
        candidate = directory / pattern.format(
            name=input_path.name, stem=input_path.stem, depth=depth
        )
        return candidate if candidate.is_file() else None
    candidates = [directory / input_path.name]
    if depth:
        candidates.extend(sorted(directory.glob(f"*depth{depth}um*.tif*")))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def save_reliability(result_name, confidence, mean_image, ground_truth, bins, epsilon):
    confidence_flat = confidence.reshape(-1)
    correct_flat = (np.abs(ground_truth.reshape(-1) - mean_image.reshape(-1)) <= epsilon).astype(
        np.int32
    )
    valid = np.isfinite(confidence_flat)
    confidence_flat, correct_flat = confidence_flat[valid], correct_flat[valid]
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(confidence_flat, edges) - 1, 0, bins - 1)
    average_confidence = np.zeros(bins, dtype=np.float64)
    accuracy = np.zeros(bins, dtype=np.float64)
    totals = np.zeros(bins, dtype=np.int64)
    for bin_index in range(bins):
        selected = indices == bin_index
        if np.any(selected):
            totals[bin_index] = int(selected.sum())
            average_confidence[bin_index] = float(confidence_flat[selected].mean())
            accuracy[bin_index] = float(correct_flat[selected].mean())
    nonempty = totals > 0
    weights = totals.astype(np.float64) / max(int(totals.sum()), 1)
    ece = float(np.sum(np.abs(average_confidence - accuracy) * weights))
    print(f"[Reliability] ECE={ece:.6f}")
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    axis.plot(average_confidence[nonempty], accuracy[nonempty], marker="o", label="Model")
    axis.set_xlabel("Average Confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title(f"Reliability Diagram (ECE={ece:.4f})")
    axis.legend()
    png_path = str(result_name).replace(".tif", "_reliability_v2.png")
    figure.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("Saved reliability diagram:", png_path)
    csv_path = png_path.replace(".png", ".csv")
    np.savetxt(
        csv_path,
        np.column_stack((average_confidence[nonempty], accuracy[nonempty], totals[nonempty])),
        delimiter=",",
        header="average_confidence,accuracy,sample_count",
        comments="",
        fmt=["%.8f", "%.8f", "%d"],
    )
    print("Saved reliability data csv:", csv_path)


def main():
    opt = parse_args()
    if not opt.bayesian:
        raise ValueError("This UQ inference script requires --bayesian true")
    if opt.mc_samples < 1:
        raise ValueError("--mc_samples must be at least one")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU
    opt.patch_y = opt.patch_x
    opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))
    opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
    opt.gap_y = opt.gap_x
    opt.ngpu = opt.GPU.count(",") + 1
    opt.batch_size = opt.batch_size or opt.ngpu
    print("\033[1;31mParameters -----> \033[0m")
    print(opt)

    model_path = Path(opt.pth_path) / opt.denoise_model
    if not model_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {model_path}")
    all_models = sorted(path for path in model_path.iterdir() if path.suffix.lower() == ".pth")
    if opt.checkpoint_slice.strip().lower() == "latest":
        model_list = all_models[-1:]
    else:
        model_list = all_models[parse_slice(opt.checkpoint_slice)]
    if not model_list:
        raise FileNotFoundError(
            f"No checkpoints selected from {model_path}; adjust --checkpoint_slice"
        )
    print([path.name for path in model_list])

    input_dir = os.path.join(opt.datasets_path, opt.datasets_folder)
    image_list = list_tiffs(input_dir)
    selected_z_test = parse_z_indices(opt.selected_z_test, len(image_list))
    print("selected_z_test", selected_z_test)
    print("\033[1;31mStacks to be processed -----> \033[0m")
    print("Total stack number -----> ", len(image_list))
    for image_path in image_list:
        print(image_path.name)

    Path(opt.output_path).mkdir(parents=True, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    output_name = (
        "DataFolderIs_"
        + Path(opt.datasets_folder).name
        + "_"
        + current_time
        + "_ModelFolderIs_"
        + opt.denoise_model
    )
    output_root = Path(opt.output_path) / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    save_test_config(opt, output_root / "para.yaml")

    device, device_ids = prepare_device(opt.GPU)
    model = maybe_data_parallel(build_uq_model(opt.patch_x, opt.patch_t, True), device, device_ids)
    if device.type == "cuda":
        print(
            "\033[1;31mUsing {} GPU(s) for testing -----> \033[0m".format(
                torch.cuda.device_count()
            )
        )

    for model_index, model_name in enumerate(model_list):
        checkpoint_dir = output_root / model_name.stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"\033[1;31mLoading model: {model_name}\033[0m")
        load_weights(model, model_name, device)
        model.eval()
        print("Model loaded successfully!")
        print(f"[INFO] bayesian={opt.bayesian}")
        snr_list = []

        for center_z in selected_z_test:
            loaded = load_uq_z_window(
                image_list,
                center_z,
                opt.patch_z,
                opt.z_edge_mode,
                opt.test_datasize,
                opt.scale_factor,
            )
            if loaded is None:
                print(f"\nSkipping stack {center_z} - not enough neighboring z layers")
                continue
            volumes, image_mean, input_dtype = loaded
            coordinates = make_uq_inference_coordinates(
                volumes[opt.patch_z // 2].shape,
                opt.patch_t,
                opt.patch_x,
                opt.gap_t,
                opt.gap_x,
            )
            loader = DataLoader(
                UQInferencePatchDataset(volumes, coordinates),
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.num_workers,
                pin_memory=device.type == "cuda",
            )
            denoise_shape = volumes[opt.patch_z // 2].shape
            result_file_name = (
                image_list[center_z].stem + "_" + model_name.stem + "_output.tif"
            )
            result_name = checkpoint_dir / result_file_name
            mean_sum = np.zeros(denoise_shape, dtype=np.float32)
            mean_square_sum = np.zeros(denoise_shape, dtype=np.float32)
            aleatoric_sum = np.zeros(denoise_shape, dtype=np.float32)
            mc_mean_images = []
            mc_sigma_images = []
            previous_time = time.time()
            time_start = time.time()

            with torch.inference_mode():
                enable_mc_dropout(model, True)
                for mc_index in range(opt.mc_samples):
                    mean_once = np.zeros(denoise_shape, dtype=np.float32)
                    aleatoric_once = np.zeros(denoise_shape, dtype=np.float32)
                    for iteration, (patch, batch_coordinates) in enumerate(loader):
                        raw_patch = patch.numpy()[:, 0, opt.patch_z // 2]
                        prediction = model(patch.to(device, non_blocking=True)).float().cpu().numpy()
                        mean_patch = prediction[:, 0, opt.patch_z // 2]
                        variance_patch = softplus_np(prediction[:, 1, opt.patch_z // 2]) + 1e-6
                        std_patch = np.sqrt(variance_patch)
                        for batch_index in range(mean_patch.shape[0]):
                            scale, slices, values = stitch_scaled_patch(
                                mean_patch,
                                raw_patch,
                                batch_coordinates,
                                batch_index,
                                mean_once,
                            )
                            aleatoric_once[slices] = std_patch[
                                batch_index,
                                values["patch_start_s"] : values["patch_end_s"],
                                values["patch_start_h"] : values["patch_end_h"],
                                values["patch_start_w"] : values["patch_end_w"],
                            ] * scale
                        elapsed = time.time() - time_start
                        batches_left = len(loader) - iteration
                        eta_seconds = int(batches_left * (time.time() - previous_time))
                        previous_time = time.time()
                        display_name = image_list[center_z].name
                        if len(display_name) > 22:
                            display_name = f"{display_name[:16]}...{display_name[-6:]}"
                        print(
                            "\r[Model %d/%d, %s] [Stack %d/%d, %s] [Patch %d/%d] "
                            "[MC %d/%d] [Time Cost: %.0d s] [ETA: %s s]     "
                            % (
                                model_index + 1,
                                len(model_list),
                                model_name.name,
                                center_z + 1,
                                len(image_list),
                                display_name,
                                iteration + 1,
                                len(loader),
                                mc_index + 1,
                                opt.mc_samples,
                                elapsed,
                                eta_seconds,
                            ),
                            end=" ",
                        )
                        if iteration + 1 == len(loader):
                            print("\n", end=" ")
                    mean_sum += mean_once
                    mean_square_sum += mean_once**2
                    aleatoric_sum += aleatoric_once
                    mc_mean_images.append(mean_once)
                    mc_sigma_images.append(aleatoric_once)

            mean_image = (mean_sum / opt.mc_samples).astype(np.float32)
            epistemic_image = np.sqrt(
                np.maximum(mean_square_sum / opt.mc_samples - mean_image**2, 0.0)
            ).astype(np.float32)
            aleatoric_image = (aleatoric_sum / opt.mc_samples).astype(np.float32)
            confidence_image = gaussian_interval_confidence(
                mc_mean_images, mc_sigma_images, opt.epsilon
            )
            save_tif_scaled(
                mean_image,
                result_name,
                "float32",
                "none",
                "none",
                opt.scale_factor,
            )
            aleatoric_name = str(result_name).replace(".tif", "_aleatoric.tif")
            epistemic_name = str(result_name).replace(".tif", "_epistemic.tif")
            confidence_name = str(result_name).replace(".tif", "_confidence.tif")
            save_tif_scaled(
                aleatoric_image,
                aleatoric_name,
                opt.save_dtype,
                opt.save_scale_mode,
                opt.tiff_compress,
                opt.scale_factor,
            )
            save_tif_scaled(
                epistemic_image,
                epistemic_name,
                opt.save_dtype,
                opt.save_scale_mode,
                opt.tiff_compress,
                opt.scale_factor,
            )
            save_tif_scaled(
                confidence_image,
                confidence_name,
                opt.save_dtype,
                opt.save_scale_mode,
                opt.tiff_compress,
                opt.scale_factor,
            )
            print("Saved UQ maps:", aleatoric_name, epistemic_name, confidence_name)
            output_image = mean_image
            print("Test result saved in:", result_name)

            gt_path = find_ground_truth(image_list[center_z], opt.gt_dir, opt.gt_pattern)
            if gt_path is not None:
                print("Automatically selected ground truth path:", gt_path)
                clean_image = tifffile.imread(gt_path).astype(np.float32)
                denoised_for_metrics = output_image.astype(np.float32)
                print("Mean of denoised image:", np.mean(denoised_for_metrics))
                print("Mean of ground truth image:", np.mean(clean_image))
                snr = calculate_snr(denoised_for_metrics, clean_image)
                print("----->SNR is", snr)
                snr_list.append(snr)
                if clean_image.shape == output_image.shape:
                    absolute_error_name = str(result_name).replace(".tif", "_abs_error.tif")
                    save_tif_scaled(
                        np.abs(output_image - clean_image),
                        absolute_error_name,
                        opt.save_dtype,
                        opt.save_scale_mode,
                        opt.tiff_compress,
                        opt.scale_factor,
                    )
                    print("Saved absolute error map:", absolute_error_name)
                    save_reliability(
                        result_name,
                        confidence_image,
                        mean_image,
                        clean_image,
                        opt.reliability_bins,
                        opt.epsilon,
                    )
                else:
                    print("[WARN] GT shape does not match output, skip absolute error map.")
                    print("[WARN] GT shape does not match mean image, skip reliability.")
            elif opt.gt_dir:
                print("[WARN] Ground truth was not found; UQ inference remains valid.")
            print(
                f"[DEBUG] saved result stats: mean={float(output_image.mean()):.6g}, "
                f"min={float(output_image.min()):.6g}, max={float(output_image.max()):.6g}"
            )

        if snr_list:
            mean_snr = np.mean(snr_list)
            print(snr_list)
            epoch_match = re.search(r"E_(\d+)", model_name.name)
            suffix = (
                f"Epoch {epoch_match.group(1)}"
                if epoch_match
                else f"model {model_index + 1}/{len(model_list)}"
            )
            print(f"--------------------------->mean SNR is {mean_snr} ({suffix})")


if __name__ == "__main__":
    main()
