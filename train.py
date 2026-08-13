"""Train SDT-4D from a directory of noisy TIFF Z slices."""

import argparse
import datetime as dt
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    build_model,
    configure_model,
    maybe_data_parallel,
    prepare_device,
    save_config,
    save_weights,
    seed_everything,
    str2bool,
    load_weights,
)
from data import TrainingPatchDataset, list_tiffs, validate_patch_args
from sampling import generate_mask_pair_4D, generate_subimages_4D


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/data/liyixin/02_real_datasets/20220905_Mouse_CD11b+Ly6G_highNA_ch2", help="Directory of Z-ordered (T,H,W) TIFF stacks")
    parser.add_argument("--output-dir", default="./pth", help="Checkpoint root directory")
    parser.add_argument("--run-name", default=None, help="Run folder name (default: data folder + timestamp)")
    parser.add_argument("--resume", default=None, help="Optional .pth checkpoint to initialize from")
    parser.add_argument("--gpu", default="4,5,6,7", help="CUDA_VISIBLE_DEVICES value, e.g. '0' or '0,1'")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--patch-x", type=int, default=128, help="Raw patch H/W; masking halves this for the model")
    parser.add_argument("--patch-t", type=int, default=64)
    parser.add_argument("--patch-z", type=int, default=5)
    parser.add_argument("--overlap-factor", type=float, default=0.25)
    parser.add_argument("--train-patches", type=int, default=6000, help="Patches sampled per epoch")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--bayesian", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ztconv-disable-residual", type=str2bool, default=False)
    parser.add_argument("--ztconv-res-scale", type=float, default=1.0)
    parser.add_argument("--decoder-skip-disable", type=str2bool, default=False)
    parser.add_argument("--decoder-skip-scale", type=float, default=1.0)
    return parser.parse_args()


def gaussian_nll(mu, target, variance, eps=1e-6):
    variance = variance.clamp_min(eps)
    return 0.5 * (torch.log(2 * math.pi * variance) + (target - mu).square() / variance).mean()


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
    seed_everything(args.seed)

    files = list_tiffs(args.data_dir)
    dataset = TrainingPatchDataset(
        files=files,
        patch_z=args.patch_z,
        patch_t=args.patch_t,
        patch_x=args.patch_x,
        overlap_factor=args.overlap_factor,
        requested_size=args.train_patches,
        max_frames=args.max_frames,
        scale_factor=args.scale_factor,
        seed=args.seed,
    )
    device, device_ids = prepare_device(args.gpu)
    if device.type != "cuda":
        raise RuntimeError("Training requires a CUDA-capable PyTorch installation and GPU")
    batch_size = args.batch_size or max(1, len(device_ids))
    args.batch_size = batch_size
    print("\033[1;31mTraining parameters -----> \033[0m")
    print(args)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.patch_x, args.patch_t, args.bayesian)
    model = maybe_data_parallel(model, device, device_ids)
    configure_model(
        model,
        ztconv_disable_residual=args.ztconv_disable_residual,
        ztconv_res_scale=args.ztconv_res_scale,
        decoder_skip_disable=args.decoder_skip_disable,
        decoder_skip_scale=args.decoder_skip_scale,
    )
    if args.resume:
        load_weights(model, args.resume, device)
        print(f"Loaded checkpoint: {args.resume}")

    param_num = sum(parameter.nelement() for parameter in model.parameters())
    print("\033[1;31mParameters of the model is {:.2f} M. \033[0m".format(param_num / 1e6))
    print("\033[1;31mUsing {} GPU(s) for training -----> \033[0m".format(len(device_ids)))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    run_name = args.run_name or f"{Path(args.data_dir).resolve().name}_{dt.datetime.now():%Y%m%d%H%M%S}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    save_config(args, run_dir / "config.yaml")

    print("\033[1;32mModel checkpoints will be saved to: {}\033[0m".format(run_dir))
    print("\033[1;32mTraining starts...\033[0m")
    prev_time = time.time()
    time_start = time.time()
    for epoch in range(1, args.epochs + 1):
        dataset.resample(epoch)
        model.train()
        for step, noisy in enumerate(loader, start=1):
            noisy = noisy.to(device, non_blocking=True)
            mask1, mask2, mask3 = generate_mask_pair_4D(noisy)
            noisy_sub1 = generate_subimages_4D(noisy, mask1)
            noisy_sub2 = generate_subimages_4D(noisy, mask2)
            noisy_sub3 = generate_subimages_4D(noisy, mask3)
            prediction = model(noisy_sub1)

            if args.bayesian:
                mean = prediction[:, :1]
                variance = F.softplus(prediction[:, 1:2]) + 1e-6
                loss1 = gaussian_nll(mean, noisy_sub2, variance)
                loss2 = gaussian_nll(mean, noisy_sub3, variance)
            else:
                loss1 = 0.5 * F.l1_loss(prediction, noisy_sub2) + 0.5 * F.mse_loss(prediction, noisy_sub2)
                loss2 = 0.5 * F.l1_loss(prediction, noisy_sub3) + 0.5 * F.mse_loss(prediction, noisy_sub3)
            loss = 0.5 * (loss1 + loss2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batches_done = (epoch - 1) * len(loader) + (step - 1)
            batches_left = args.epochs * len(loader) - batches_done
            time_left = dt.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
            prev_time = time.time()
            time_end = time.time()
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f] [ETA: %s] [Time cost: %.2d s] "
                % (
                    epoch,
                    args.epochs,
                    step,
                    len(loader),
                    loss,
                    time_left,
                    time_end - time_start,
                ),
                end=" ",
            )
        print("\n", end=" ")
        checkpoint = run_dir / f"E_{epoch:03d}_Iter_{len(loader):04d}.pth"
        save_weights(model, checkpoint)


if __name__ == "__main__":
    main()
