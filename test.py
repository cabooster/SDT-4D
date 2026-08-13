"""Run SDT-4D inference, optionally evaluating against ground truth."""

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import re
import time

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader

from common import (
    build_model,
    configure_model,
    load_weights,
    maybe_data_parallel,
    prepare_device,
    save_config,
    str2bool,
)
from data import (
    InferencePatchDataset,
    list_tiffs,
    load_z_window,
    make_inference_coordinates,
    validate_patch_args,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/data/liyixin/02_real_datasets/20220905_Mouse_CD11b+Ly6G_highNA_ch2", help="Directory of Z-ordered noisy (T,H,W) TIFF stacks")
    parser.add_argument(
        "--checkpoint",
        default="20220905_Mouse_CD11b+Ly6G_highNA_ch2_20260810194346",
        help="Model folder name under ./pth, or an explicit .pth file/directory path",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        default=False,
        help="Test every .pth in a checkpoint directory",
    )
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--gt-dir", default=None, help="Optional ground-truth TIFF directory; omit for inference only")
    parser.add_argument(
        "--gt-pattern",
        default=None,
        help="Optional GT filename template with {name}, {stem}, or {depth}, e.g. clean_depth{depth}um.tif",
    )
    parser.add_argument("--gpu", default="4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-x", type=int, default=64)
    parser.add_argument("--patch-t", type=int, default=64)
    parser.add_argument("--patch-z", type=int, default=5)
    parser.add_argument("--overlap-factor", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--bayesian", type=str2bool, default=False)
    parser.add_argument("--z-edge-mode", choices=("skip", "replicate", "reflect"), default="replicate")
    parser.add_argument(
        "--z-indices",
        default="all",
        help="Centers to process: 'all', comma list, or inclusive ranges such as 0,2,5-8",
    )
    parser.add_argument("--ztconv-disable-z", type=str2bool, default=False)
    parser.add_argument("--ztconv-disable-thw", type=str2bool, default=False)
    parser.add_argument("--ztconv-z-scale", type=float, default=1.0)
    parser.add_argument("--ztconv-thw-scale", type=float, default=1.0)
    parser.add_argument("--ztconv-disable-residual", type=str2bool, default=False)
    parser.add_argument("--ztconv-res-scale", type=float, default=1.0)
    parser.add_argument("--decoder-skip-disable", type=str2bool, default=False)
    parser.add_argument("--decoder-skip-scale", type=float, default=1.0)
    return parser.parse_args()


def resolve_checkpoints(path, all_checkpoints):
    path_text = os.path.expanduser(str(path))
    path = Path(path_text)
    if (
        not path.is_absolute()
        and path.parent == Path(".")
        and not path_text.startswith(("./", "../"))
    ):
        path = Path("./pth") / path
    path = path.resolve()
    if path.is_file() and path.suffix.lower() == ".pth":
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoints = sorted(path.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pth checkpoints found in: {path}")
    return checkpoints if all_checkpoints else [checkpoints[-1]]


def parse_z_indices(spec, count):
    if spec.strip().lower() == "all":
        return list(range(count))
    indices = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(value) for value in item.split("-", 1))
            indices.update(range(start, end + 1))
        else:
            indices.add(int(item))
    invalid = sorted(index for index in indices if not 0 <= index < count)
    if invalid:
        raise ValueError(f"Z indices outside [0, {count - 1}]: {invalid}")
    if not indices:
        raise ValueError("--z-indices did not select any Z slices")
    return sorted(indices)


def _batch_value(coordinates, key, index):
    value = coordinates[key]
    if torch.is_tensor(value):
        return int(value[index].item())
    return int(value[index])


def stitch_batch(output, coordinates, destination):
    for index in range(output.shape[0]):
        st0 = _batch_value(coordinates, "stack_start_s", index)
        st1 = _batch_value(coordinates, "stack_end_s", index)
        sh0 = _batch_value(coordinates, "stack_start_h", index)
        sh1 = _batch_value(coordinates, "stack_end_h", index)
        sw0 = _batch_value(coordinates, "stack_start_w", index)
        sw1 = _batch_value(coordinates, "stack_end_w", index)
        pt0 = _batch_value(coordinates, "patch_start_s", index)
        pt1 = _batch_value(coordinates, "patch_end_s", index)
        ph0 = _batch_value(coordinates, "patch_start_h", index)
        ph1 = _batch_value(coordinates, "patch_end_h", index)
        pw0 = _batch_value(coordinates, "patch_start_w", index)
        pw1 = _batch_value(coordinates, "patch_end_w", index)
        destination[st0:st1, sh0:sh1, sw0:sw1] = output[index, pt0:pt1, ph0:ph1, pw0:pw1]


def save_like_input(path, image, source_dtype, bayesian):
    source_dtype = np.dtype(source_dtype)
    if bayesian or np.issubdtype(source_dtype, np.floating):
        tifffile.imwrite(path, image.astype(np.float32))
        return
    limits = np.iinfo(source_dtype)
    tifffile.imwrite(path, np.clip(image, limits.min, limits.max).astype(source_dtype))


def find_ground_truth(input_path, gt_dir, pattern):
    if gt_dir is None:
        return None
    gt_dir = Path(gt_dir).expanduser().resolve()
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")
    depth_match = re.search(r"depth(\d+)um", input_path.name, flags=re.IGNORECASE)
    depth = depth_match.group(1) if depth_match else ""
    if pattern:
        candidate = gt_dir / pattern.format(name=input_path.name, stem=input_path.stem, depth=depth)
        return candidate if candidate.is_file() else None

    candidates = [gt_dir / input_path.name]
    replaced = re.sub("noisy", "clean", input_path.name, flags=re.IGNORECASE)
    candidates.append(gt_dir / replaced)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if depth:
        matches = sorted(path for path in gt_dir.glob("*.tif*") if f"depth{depth}um" in path.name)
        if len(matches) == 1:
            return matches[0]
    return None


def calculate_snr(estimate, ground_truth):
    if estimate.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: output {estimate.shape}, GT {ground_truth.shape}")
    signal_energy = np.sum(np.square(ground_truth, dtype=np.float64))
    noise_energy = np.sum(np.square(estimate - ground_truth, dtype=np.float64))
    if noise_energy == 0:
        return float("inf")
    if signal_energy == 0:
        return float("-inf")
    return float(10 * np.log10(signal_energy / noise_energy))


def main():
    args = parse_args()
    validate_patch_args(
        args.patch_z,
        args.patch_t,
        args.patch_x,
        args.overlap_factor,
        args.scale_factor,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    input_files = list_tiffs(args.input_dir)
    checkpoints = resolve_checkpoints(args.checkpoint, args.all_checkpoints)
    z_indices = parse_z_indices(args.z_indices, len(input_files))
    device, device_ids = prepare_device(args.gpu)
    batch_size = args.batch_size or max(1, len(device_ids))
    args.batch_size = batch_size
    print("\033[1;31mParameters -----> \033[0m")
    print(args)
    print([checkpoint.name for checkpoint in checkpoints])
    print("selected_z_test", z_indices)
    print("\033[1;31mStacks to be processed -----> \033[0m")
    print("Total stack number -----> ", len(input_files))
    for input_file in input_files:
        print(input_file.name)

    model = build_model(args.patch_x, args.patch_t, args.bayesian)
    model = maybe_data_parallel(model, device, device_ids)
    if device.type == "cuda":
        print("\033[1;31mUsing {} GPU(s) for testing -----> \033[0m".format(len(device_ids)))
    configure_model(
        model,
        ztconv_disable_z=args.ztconv_disable_z,
        ztconv_disable_thw=args.ztconv_disable_thw,
        ztconv_z_scale=args.ztconv_z_scale,
        ztconv_thw_scale=args.ztconv_thw_scale,
        ztconv_disable_residual=args.ztconv_disable_residual,
        ztconv_res_scale=args.ztconv_res_scale,
        decoder_skip_disable=args.decoder_skip_disable,
        decoder_skip_scale=args.decoder_skip_scale,
    )

    checkpoint_input = Path(os.path.expanduser(str(args.checkpoint)))
    if checkpoint_input.suffix.lower() == ".pth":
        model_name = checkpoint_input.parent.name or checkpoint_input.stem
    else:
        model_name = checkpoint_input.name
    output_name = (
        f"DataFolderIs_{Path(args.input_dir).expanduser().resolve().name}_"
        f"{dt.datetime.now():%Y%m%d%H%M}_ModelFolderIs_{model_name}"
    )
    output_root = Path(args.output_dir).expanduser().resolve() / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(args, output_root / "inference_config.yaml")

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        print(f"\033[1;31mLoading model: {checkpoint}\033[0m")
        load_weights(model, checkpoint, device)
        model.eval()
        print("Model loaded successfully!")
        print(f"[INFO] bayesian={args.bayesian}")
        checkpoint_dir = output_root / checkpoint.stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metric_rows = []

        for position, center_z in enumerate(z_indices, start=1):
            loaded = load_z_window(
                input_files,
                center_z,
                args.patch_z,
                args.z_edge_mode,
                args.max_frames,
                args.scale_factor,
            )
            if loaded is None:
                print(f"\nSkipping stack {center_z} - not enough neighboring z layers")
                continue
            volumes, center_mean, source_dtype = loaded
            gap_t = max(1, int(args.patch_t * (1 - args.overlap_factor)))
            gap_x = max(1, int(args.patch_x * (1 - args.overlap_factor)))
            coordinates = make_inference_coordinates(
                volumes[args.patch_z // 2].shape,
                args.patch_t,
                args.patch_x,
                gap_t,
                gap_x,
            )
            loader = DataLoader(
                InferencePatchDataset(volumes, coordinates),
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            denoised = np.zeros_like(volumes[args.patch_z // 2], dtype=np.float32)
            prev_time = time.time()
            time_start = time.time()
            with torch.inference_mode():
                for patch_index, (patches, batch_coordinates) in enumerate(loader):
                    prediction = model(patches.to(device, non_blocking=True))
                    if args.bayesian:
                        prediction = prediction[:, :1]
                    center = prediction[:, 0, args.patch_z // 2].float().cpu().numpy()
                    stitch_batch(center, batch_coordinates, denoised)

                    batches_left = len(loader) - patch_index
                    time_left_seconds = int(batches_left * (time.time() - prev_time))
                    prev_time = time.time()
                    time_cost = time.time() - time_start
                    display_name = input_files[center_z].name
                    if len(display_name) > 22:
                        display_name = f"{display_name[:16]}...{display_name[-6:]}"
                    print(
                        "\r[Model %d/%d, %s] [Stack %d/%d, %s] [Patch %d/%d] "
                        "[Time Cost: %.0d s] [ETA: %s s]     "
                        % (
                            checkpoint_index + 1,
                            len(checkpoints),
                            checkpoint.name,
                            center_z + 1,
                            len(input_files),
                            display_name,
                            patch_index + 1,
                            len(loader),
                            time_cost,
                            time_left_seconds,
                        ),
                        end=" ",
                    )
                    if patch_index + 1 == len(loader):
                        print("\n", end=" ")

            output = (denoised + center_mean) * args.scale_factor
            output_path = checkpoint_dir / f"{input_files[center_z].stem}_denoised.tif"
            save_like_input(output_path, output, source_dtype, args.bayesian)
            print("Test result saved in:", output_path)

            gt_path = find_ground_truth(input_files[center_z], args.gt_dir, args.gt_pattern)
            if gt_path is not None:
                print("Automatically selected ground truth path:", gt_path)
                ground_truth = tifffile.imread(gt_path).astype(np.float32)
                snr = calculate_snr(output.astype(np.float32), ground_truth)
                print("Mean of denoised image:", np.mean(output))
                print("Mean of ground truth image:", np.mean(ground_truth))
                print("----->SNR is", snr)
                metric_rows.append((output_path.name, str(gt_path), snr))
            elif args.gt_dir:
                print("[WARN] Ground truth not found; inference result is still valid.")

            print(
                f"[DEBUG] saved result stats: mean={float(output.mean()):.6g}, "
                f"min={float(output.min()):.6g}, max={float(output.max()):.6g}"
            )

        if metric_rows:
            snr_values = [row[2] for row in metric_rows]
            mean_snr = float(np.mean(snr_values))
            print("\033[1;31m" + str(snr_values) + "\033[0m")
            epoch_match = re.search(r"E_(\d+)", checkpoint.name)
            suffix = f"Epoch {epoch_match.group(1)}" if epoch_match else f"model {checkpoint_index + 1}/{len(checkpoints)}"
            print("\033[1;31m" + f"--------------------------->mean SNR is {mean_snr} ({suffix})" + "\033[0m")
            metrics_path = checkpoint_dir / "metrics.csv"
            with metrics_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("result", "ground_truth", "snr_db"))
                writer.writerows(metric_rows)
                writer.writerow(("mean", "", mean_snr))


if __name__ == "__main__":
    main()
