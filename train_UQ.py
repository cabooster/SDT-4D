"""Train the SDT-4D UQ model with staged mean and NLL objectives."""

import argparse
import datetime
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
import yaml

from common import maybe_data_parallel, prepare_device, save_weights, str2bool
from sampling import generate_mask_pair_4D, generate_subimages_4D
from SDT4D import SDT4D
from uq_data import UQTrainingPatchDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument(
        "--GPU",
        type=str,
        default="4,5,6,7",
        help="GPU indices used for computation, for example '0' or '0,1,2'",
    )
    parser.add_argument("--patch_x", type=int, default=128, help="Patch size in X and Y")
    parser.add_argument("--patch_t", type=int, default=64, help="Patch size in T")
    parser.add_argument("--patch_z", type=int, default=5, help="Patch size in Z")
    parser.add_argument(
        "--overlap_factor",
        type=float,
        default=0.25,
        help="Overlap factor between adjacent patches",
    )
    parser.add_argument(
        "--train_datasets_size",
        type=int,
        default=6000,
        help="Target number of patches used to determine temporal sampling",
    )
    parser.add_argument("--datasets_path", type=str, default="", help="Dataset root path")
    parser.add_argument("--pth_path", type=str, default="./pth", help="Model root path")
    parser.add_argument(
        "--datasets_folder",
        type=str,
        default="/data/liyixin/01_noisy_datasets/noise_2RPN_alpha_beta",
        help="Folder containing training TIFF stacks",
    )
    parser.add_argument("--output_path", type=str, default="./results", help="Output directory")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument(
        "--select_img_num",
        type=int,
        default=10000000000,
        help="Maximum number of frames used for training",
    )
    parser.add_argument(
        "--test_datasize",
        type=int,
        default=10000000000,
        help="Maximum number of frames used for testing",
    )
    parser.add_argument("--scale_factor", type=float, default=1, help="Image intensity scale factor")
    parser.add_argument(
        "--pg_enable",
        type=str2bool,
        default=True,
        help="Use Poisson-Gaussian observation variance inside the NLL",
    )
    parser.add_argument("--pg_alpha", type=float, default=2.0, help="Initial PG alpha")
    parser.add_argument("--pg_beta", type=float, default=2.0, help="Initial PG beta variance")
    parser.add_argument(
        "--pg_detach_mu",
        type=str2bool,
        default=True,
        help="Detach the mean when forming PG observation variance",
    )
    parser.add_argument("--pg_var_scale", type=float, default=1.0, help="Base PG variance scale")
    parser.add_argument(
        "--bayesian",
        type=str2bool,
        default=True,
        help="Enable aleatoric mean and variance output",
    )
    parser.add_argument("--stage1_epochs", type=int, default=10, help="Stage 1 epoch count")
    parser.add_argument("--lambda_mu_start", type=float, default=0.8)
    parser.add_argument("--lambda_mu_end", type=float, default=0.05)
    parser.add_argument("--pg_scale_start", type=float, default=0.5)
    parser.add_argument("--pg_scale_end", type=float, default=1.0)
    parser.add_argument("--pg_ramp_end_epoch", type=int, default=40)
    parser.add_argument("--nll_ramp_end_epoch", type=int, default=-1)
    parser.add_argument("--mu_l2_w_stage1_start", type=float, default=0.5)
    parser.add_argument("--mu_l2_w_stage1_end", type=float, default=0.2)
    parser.add_argument("--mu_l2_w_stage2", type=float, default=0.1)
    parser.add_argument(
        "--pg_learn_ab",
        type=str2bool,
        default=True,
        help="Learn PG alpha and beta during Stage 2",
    )
    parser.add_argument(
        "--pg_ab_lr_mult",
        type=float,
        default=5,
        help="Learning-rate multiplier for alpha and beta",
    )
    parser.add_argument(
        "--pg_ab_init_mode",
        choices=("fixed", "zeros", "from_stage1"),
        default="zeros",
        help="Alpha and beta initialization strategy",
    )
    parser.add_argument(
        "--pg_ab_calib_steps",
        type=int,
        default=200,
        help="Mini-batches used for Stage 2 regression initialization",
    )
    parser.add_argument(
        "--pg_mu_nonneg",
        type=str2bool,
        default=True,
        help="Clamp the mean to nonnegative values when forming PG variance",
    )
    parser.add_argument(
        "--var_tot_max",
        type=float,
        default=0.0,
        help="Optional maximum total variance; zero disables the limit",
    )
    parser.add_argument("--num_workers", type=int, default=0)
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


def save_training_config(opt, path):
    """Write the same training fields as the original UQ entry point."""

    fields = (
        "n_epochs",
        "GPU",
        "batch_size",
        "datasets_folder",
        "datasets_path",
        "output_path",
        "pth_path",
        "patch_x",
        "patch_y",
        "patch_t",
        "gap_y",
        "gap_x",
        "gap_t",
        "lr",
        "b1",
        "b2",
        "select_img_num",
        "train_datasets_size",
        "bayesian",
    )
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump({field: getattr(opt, field) for field in fields}, file, sort_keys=False)


def gaussian_nll_with_total_var(mu, target, var_tot, reduction="mean", eps=1e-6):
    var_tot = var_tot.clamp_min(eps)
    residual_squared = (target - mu) ** 2
    loss_mu = 0.5 * residual_squared / var_tot.detach()
    loss_var = 0.5 * (
        torch.log(2 * math.pi * var_tot) + residual_squared.detach() / var_tot
    )
    nll = loss_mu + loss_var
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    raise ValueError("Invalid reduction")


def linear_ramp(current, start_epoch, end_epoch):
    if current < start_epoch:
        return 0.0
    if end_epoch <= start_epoch:
        return 1.0
    progress = float(current - start_epoch) / float(end_epoch - start_epoch)
    return float(min(1.0, max(0.0, progress)))


def compute_stage_schedules(epoch, opt):
    if epoch < opt.stage1_epochs:
        fraction = 1.0 if opt.stage1_epochs <= 1 else 1.0 - epoch / (opt.stage1_epochs - 1)
        lambda_mu = opt.lambda_mu_end + (opt.lambda_mu_start - opt.lambda_mu_end) * fraction
        l2_weight = opt.mu_l2_w_stage1_start + (
            opt.mu_l2_w_stage1_end - opt.mu_l2_w_stage1_start
        ) * (1.0 - fraction)
        return True, float(lambda_mu), 0.0, 0.0, float(l2_weight)

    ramp_end = opt.pg_ramp_end_epoch if opt.pg_ramp_end_epoch >= 0 else opt.n_epochs - 1
    ramp_end = max(ramp_end, opt.stage1_epochs)
    pg_progress = linear_ramp(epoch, opt.stage1_epochs, ramp_end)
    pg_scale = opt.pg_scale_start + (opt.pg_scale_end - opt.pg_scale_start) * pg_progress
    nll_end = opt.nll_ramp_end_epoch if opt.nll_ramp_end_epoch >= 0 else ramp_end
    nll_blend = linear_ramp(epoch, opt.stage1_epochs, max(nll_end, opt.stage1_epochs))
    return False, float(opt.lambda_mu_end), float(pg_scale), float(nll_blend), float(
        opt.mu_l2_w_stage2
    )


@torch.no_grad()
def calibrate_ab_from_stream(model, loader, opt, alpha_raw, beta_raw, device):
    """Initialize PG parameters with robust binned regression on view differences."""

    model.eval()
    total_weight = sum_x = sum_y = sum_xx = sum_xy = 0.0
    for step, noisy in enumerate(loader):
        if step >= min(opt.pg_ab_calib_steps, len(loader)):
            break
        noisy = noisy.to(device, non_blocking=True)
        mask1, mask2, mask3 = generate_mask_pair_4D(noisy)
        noisy_sub1 = generate_subimages_4D(noisy, mask1)
        noisy_sub2 = generate_subimages_4D(noisy, mask2)
        noisy_sub3 = generate_subimages_4D(noisy, mask3)
        mean = model(noisy_sub1)[:, 0:1]
        x = (mean.clamp_min(0.0) if opt.pg_mu_nonneg else mean).reshape(-1).float()
        y = (0.5 * (noisy_sub2 - noisy_sub3) ** 2).reshape(-1).float()
        count = min(300000, x.numel())
        indices = torch.randint(0, x.numel(), (count,), device=x.device)
        x, y = x[indices], y[indices]
        keep = y <= y.quantile(0.98)
        x, y = x[keep], y[keep]
        if x.numel() < 2:
            continue
        edges = torch.linspace(float(x.min()), float(x.max()) + 1e-8, steps=16, device=x.device)
        bins = torch.bucketize(x, edges) - 1
        for bin_index in range(15):
            selected = bins == bin_index
            if selected.any():
                weight = float(selected.sum().item())
                x_value = float(x[selected].mean().item())
                y_value = float(y[selected].median().item())
                total_weight += weight
                sum_x += weight * x_value
                sum_y += weight * y_value
                sum_xx += weight * x_value * x_value
                sum_xy += weight * x_value * y_value
    denominator = total_weight * sum_xx - sum_x * sum_x
    if denominator > 0:
        alpha = max(0.0, (total_weight * sum_xy - sum_x * sum_y) / denominator)
        beta = max(0.0, (sum_y - alpha * sum_x) / total_weight)
        alpha_raw.copy_(torch.log(torch.tensor(alpha + 1e-8, device=device)))
        beta_raw.copy_(torch.log(torch.tensor(beta + 1e-8, device=device)))
    model.train()


def main():
    opt = parse_args()
    if not opt.bayesian:
        raise ValueError("This UQ training script requires --bayesian true")
    if opt.patch_z < 1 or opt.patch_z % 2 == 0:
        raise ValueError("--patch_z must be a positive odd number")
    if opt.patch_x < 2 or opt.patch_x % 2:
        raise ValueError("--patch_x must be a positive even number")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU
    opt.patch_y = opt.patch_x
    opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
    opt.gap_y = opt.gap_x
    opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))
    opt.ngpu = opt.GPU.count(",") + 1
    opt.batch_size = opt.ngpu
    print("\033[1;31mTraining parameters -----> \033[0m")
    print(opt)

    Path(opt.output_path).mkdir(parents=True, exist_ok=True)
    current_time = Path(opt.datasets_folder).name + "_" + datetime.datetime.now().strftime(
        "%Y%m%d%H%M"
    )
    current_time += "_bayesian_mu2nll_smooth"
    print(
        "\033[1;32mUsing Bayesian training with smooth transition from "
        "mu(L1+L2) to NLL+PG.\033[0m"
    )
    output_path = os.path.join(opt.output_path, current_time)
    del output_path
    pth_path = os.path.join("pth", current_time)
    print("ckp is saved in {}".format(pth_path))
    Path(pth_path).mkdir(parents=True, exist_ok=True)

    data_dir = os.path.join(opt.datasets_path, opt.datasets_folder)
    dataset = UQTrainingPatchDataset(
        data_dir=data_dir,
        patch_z=opt.patch_z,
        patch_t=opt.patch_t,
        patch_x=opt.patch_x,
        overlap_factor=opt.overlap_factor,
        requested_size=opt.train_datasets_size,
        max_frames=opt.select_img_num,
        scale_factor=opt.scale_factor,
    )
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    save_training_config(opt, Path(pth_path) / "para.yaml")

    device, device_ids = prepare_device(opt.GPU)
    if device.type != "cuda":
        raise RuntimeError("Training requires a CUDA-capable PyTorch installation and GPU")
    model = build_uq_model(opt.patch_x, opt.patch_t, True).to(device)
    summary(
        model,
        input_size=(1, opt.patch_z, opt.patch_t, opt.patch_x // 2, opt.patch_x // 2),
        device="cuda",
    )
    model = maybe_data_parallel(model, device, device_ids)
    parameter_count = sum(parameter.nelement() for parameter in model.parameters())
    print("\033[1;31mParameters of the model is {:.2f} M. \033[0m".format(parameter_count / 1e6))
    print(
        "\033[1;31mUsing {} GPU(s) for training -----> \033[0m".format(
            torch.cuda.device_count()
        )
    )
    print("Pretrained model not found, training from scratch.")

    if opt.pg_learn_ab:
        alpha_start = max(opt.pg_alpha, 1e-8)
        beta_start = max(opt.pg_beta, 1e-8)
        if opt.pg_ab_init_mode == "zeros":
            alpha_start, beta_start = 2.0, 2.0
        alpha_raw = nn.Parameter(torch.log(torch.tensor(alpha_start, device=device)))
        beta_raw = nn.Parameter(torch.log(torch.tensor(beta_start, device=device)))
        learnable_ab = [alpha_raw, beta_raw]
    else:
        alpha_raw = beta_raw = None
        learnable_ab = []
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": learnable_ab, "lr": opt.lr * opt.pg_ab_lr_mult},
        ],
        lr=opt.lr,
        betas=(opt.b1, opt.b2),
    )
    l1_loss = nn.L1Loss().to(device)
    l2_loss = nn.MSELoss().to(device)
    previous_time = time.time()
    time_start = time.time()
    calibrated = False

    for epoch in range(opt.n_epochs):
        if (
            opt.pg_learn_ab
            and not calibrated
            and epoch >= opt.stage1_epochs
            and opt.pg_ab_init_mode == "from_stage1"
        ):
            calibrate_ab_from_stream(model, loader, opt, alpha_raw, beta_raw, device)
            calibrated = True
        model.train()
        for iteration, noisy in enumerate(loader):
            noisy = noisy.to(device, non_blocking=True)
            mask1, mask2, mask3 = generate_mask_pair_4D(noisy)
            noisy_sub1 = generate_subimages_4D(noisy, mask1)
            noisy_sub2 = generate_subimages_4D(noisy, mask2)
            noisy_sub3 = generate_subimages_4D(noisy, mask3)
            prediction = model(noisy_sub1)
            in_stage1, lambda_mu, pg_scale, nll_blend, l2_weight = compute_stage_schedules(
                epoch, opt
            )
            mean = prediction[:, 0:1]
            raw_variance = prediction[:, 1:2]
            loss_mu_l1 = 0.5 * l1_loss(mean, noisy_sub2) + 0.5 * l1_loss(mean, noisy_sub3)
            loss_mu_l2 = 0.5 * l2_loss(mean, noisy_sub2) + 0.5 * l2_loss(mean, noisy_sub3)
            if in_stage1:
                nll_half = torch.tensor(0.0, device=device, dtype=mean.dtype)
                total_loss = lambda_mu * loss_mu_l1 + l2_weight * loss_mu_l2
            else:
                clean_variance = F.softplus(raw_variance) + 1e-6
                if opt.pg_enable:
                    if opt.pg_learn_ab:
                        alpha, beta = torch.exp(alpha_raw), torch.exp(beta_raw)
                    else:
                        alpha = torch.as_tensor(max(opt.pg_alpha, 0.0), device=device)
                        beta = torch.as_tensor(max(opt.pg_beta, 0.0), device=device)
                    noise_mean = mean.detach() if opt.pg_detach_mu else mean
                    if opt.pg_mu_nonneg:
                        noise_mean = noise_mean.clamp_min(0.0)
                    noise_variance = (
                        alpha * noise_mean + beta
                    ) * opt.pg_var_scale * pg_scale
                    total_variance = clean_variance + noise_variance
                else:
                    total_variance = clean_variance
                total_variance = total_variance.clamp_min(1e-6)
                if opt.var_tot_max > 0:
                    total_variance = total_variance.clamp_max(opt.var_tot_max)
                nll_half = 0.5 * (
                    gaussian_nll_with_total_var(mean, noisy_sub2, total_variance)
                    + gaussian_nll_with_total_var(mean, noisy_sub3, total_variance)
                )
                total_loss = (
                    lambda_mu * loss_mu_l1 + l2_weight * loss_mu_l2 + nll_blend * nll_half
                )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            batches_done = epoch * len(loader) + iteration
            batches_left = opt.n_epochs * len(loader) - batches_done
            time_left = datetime.timedelta(
                seconds=int(batches_left * (time.time() - previous_time))
            )
            previous_time = time.time()
            del time_left
            alpha_value = (
                float(torch.exp(alpha_raw).detach().item())
                if opt.pg_enable and opt.pg_learn_ab
                else float(max(opt.pg_alpha, 0.0)) if opt.pg_enable else 0.0
            )
            beta_value = (
                float(torch.exp(beta_raw).detach().item())
                if opt.pg_enable and opt.pg_learn_ab
                else float(max(opt.pg_beta, 0.0)) if opt.pg_enable else 0.0
            )
            print(
                "\r[E %d/%d][%s][B %d/%d][TL %.6f][N %.6f][M(L1) %.6f]"
                "[M(L2) %.6f][lm %.3f][pg %.2f][nb %.2f][a %.2f][b %.2f]"
                % (
                    epoch + 1,
                    opt.n_epochs,
                    "S1" if in_stage1 else "S2",
                    iteration + 1,
                    len(loader),
                    float(total_loss),
                    float(nll_half),
                    float(loss_mu_l1),
                    float(loss_mu_l2),
                    lambda_mu,
                    pg_scale,
                    nll_blend,
                    alpha_value,
                    beta_value,
                ),
                end=" ",
            )
            if iteration + 1 == len(loader):
                print("\n", end=" ")
                filename = (
                    "E_"
                    + str(epoch + 1).zfill(3)
                    + "_Iter_"
                    + str(iteration + 1).zfill(4)
                    + ".pth"
                )
                save_weights(model, Path(pth_path) / filename)
                log_path = Path(pth_path) / "ab_values.txt"
                if not log_path.exists():
                    log_path.write_text("epoch\ta\tb\n", encoding="utf-8")
                with log_path.open("a", encoding="utf-8") as file:
                    file.write(f"{epoch + 1}\t{alpha_value:.6f}\t{beta_value:.6f}\n")
        _ = time.time() - time_start


if __name__ == "__main__":
    main()
