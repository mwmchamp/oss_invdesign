"""Train the S-parameter surrogate CNN.

Loss function: weighted combination of
  1. MSE on normalized real/imag (captures absolute errors)
  2. MSE on log-magnitude (captures relative errors across dynamic range)
  3. Phase error (ensures phase accuracy for well-coupled ports)

Balanced sampling: when combining datasets with different fill distributions,
uses WeightedRandomSampler to equalize sparse/mid/dense fill representation.

Stratified evaluation: reports per-fill-bin metrics to verify uniform accuracy.

Usage (quick test on 500 designs):
    python surrogate/train.py --max-designs 500 --epochs 100

Usage (full training):
    python surrogate/train.py --epochs 300 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler

from surrogate.data import (
    PixelGridDataset,
    upper_tri_to_sparams,
    FILL_BIN_NAMES,
)
from surrogate.model import SParamCNN, SParamCNNv2, SParamResNet, SParamEM300


def log_mag_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6,
                 weights: torch.Tensor | None = None) -> torch.Tensor:
    """MSE on log10(|S|) — equalizes dynamic range between S11≈1 and S21≈0.001."""
    pred_mag = torch.sqrt(pred[..., 0] ** 2 + pred[..., 1] ** 2 + eps)
    target_mag = torch.sqrt(target[..., 0] ** 2 + target[..., 1] ** 2 + eps)
    per_sample = (torch.log10(pred_mag) - torch.log10(target_mag)).pow(2).mean(
        dim=list(range(1, pred_mag.ndim)))  # (B,)
    if weights is not None:
        return (per_sample * weights).sum() / weights.sum()
    return per_sample.mean()


def phase_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6,
               weights: torch.Tensor | None = None) -> torch.Tensor:
    """Cosine-distance loss on phase — avoids phase-wrapping issues."""
    pred_cpx = torch.complex(pred[..., 0], pred[..., 1])
    target_cpx = torch.complex(target[..., 0], target[..., 1])

    pred_mag = pred_cpx.abs() + eps
    target_mag = target_cpx.abs() + eps

    pred_norm = pred_cpx / pred_mag
    target_norm = target_cpx / target_mag

    mask = target_mag > 0.01
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    cos_diff = (pred_norm * target_norm.conj()).real
    per_sample_loss = 1.0 - cos_diff  # (B, F, 10)
    if weights is not None:
        # Apply mask then weight per sample
        per_sample_loss = per_sample_loss * mask.float()
        per_sample = per_sample_loss.mean(dim=list(range(1, per_sample_loss.ndim)))  # (B,)
        return (per_sample * weights).sum() / weights.sum()
    return (1.0 - cos_diff[mask]).mean()


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-sample weighted MSE. weights: (B,), pred/target: (B, ...)."""
    per_sample = (pred - target).pow(2).mean(dim=list(range(1, pred.ndim)))  # (B,)
    return (per_sample * weights).sum() / weights.sum()


def _weighted_reco_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    kind: str,
    huber_delta: float = 0.1,
) -> torch.Tensor:
    """Per-sample weighted reconstruction loss on normalized real/imag.

    kind ∈ {"mse", "mae", "huber"}. Karahan et al. JSSC 2023 / Nat. Commun.
    2024 use MAE; MAE spreads capacity across the dynamic range better than
    MSE, which over-weights the dense-fill regime where errors are small.
    """
    diff = pred - target
    if kind == "mse":
        elem = diff.pow(2)
    elif kind == "mae":
        elem = diff.abs()
    elif kind == "huber":
        absd = diff.abs()
        elem = torch.where(absd < huber_delta,
                           0.5 * diff.pow(2) / huber_delta,
                           absd - 0.5 * huber_delta)
    else:
        raise ValueError(f"Unknown loss kind: {kind}")
    per_sample = elem.mean(dim=list(range(1, pred.ndim)))  # (B,)
    return (per_sample * weights).sum() / weights.sum()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dataset: PixelGridDataset,
    w_mse: float = 1.0,
    w_logmag: float = 0.5,
    w_phase: float = 0.1,
    loss_kind: str = "mae",
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_logmag = 0.0
    total_phase = 0.0
    n = 0

    target_mean = torch.from_numpy(dataset.target_mean).to(device)
    target_std = torch.from_numpy(dataset.target_std).to(device)

    for grids, targets, cw in loader:
        grids = grids.to(device)
        targets = targets.to(device)
        cw = cw.to(device)  # (B,) coupling weights

        pred_norm = model(grids)
        mse = _weighted_reco_loss(pred_norm, targets, cw, kind=loss_kind)

        pred_raw = pred_norm * target_std + target_mean
        target_raw = targets * target_std + target_mean

        lm = log_mag_loss(pred_raw, target_raw, weights=cw)
        ph = phase_loss(pred_raw, target_raw, weights=cw)

        loss = w_mse * mse + w_logmag * lm + w_phase * ph

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = grids.shape[0]
        total_loss += loss.item() * bs
        total_mse += mse.item() * bs
        total_logmag += lm.item() * bs
        total_phase += ph.item() * bs
        n += bs

    return {
        "loss": total_loss / n,
        "mse": total_mse / n,
        "logmag": total_logmag / n,
        "phase": total_phase / n,
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset: PixelGridDataset,
) -> dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_logmag = 0.0
    total_mag_err = 0.0
    n = 0

    target_mean = torch.from_numpy(dataset.target_mean).to(device)
    target_std = torch.from_numpy(dataset.target_std).to(device)

    for batch in loader:
        grids, targets = batch[0], batch[1]
        grids = grids.to(device)
        targets = targets.to(device)

        pred_norm = model(grids)
        mse = nn.functional.mse_loss(pred_norm, targets)

        pred_raw = pred_norm * target_std + target_mean
        target_raw = targets * target_std + target_mean

        lm = log_mag_loss(pred_raw, target_raw)

        eps = 1e-6
        pred_mag_db = 20 * torch.log10(torch.sqrt(pred_raw[..., 0]**2 + pred_raw[..., 1]**2 + eps))
        target_mag_db = 20 * torch.log10(torch.sqrt(target_raw[..., 0]**2 + target_raw[..., 1]**2 + eps))
        mag_err_db = (pred_mag_db - target_mag_db).abs().mean()

        bs = grids.shape[0]
        total_mse += mse.item() * bs
        total_logmag += lm.item() * bs
        total_mag_err += mag_err_db.item() * bs
        n += bs

    return {
        "mse": total_mse / n,
        "logmag": total_logmag / n,
        "mag_err_db": total_mag_err / n,
    }


@torch.no_grad()
def eval_stratified(
    model: nn.Module,
    dataset: PixelGridDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, dict[str, float]]:
    """Evaluate per-fill-bin on a subset of the dataset."""
    model.eval()
    target_mean = torch.from_numpy(dataset.target_mean).to(device)
    target_std = torch.from_numpy(dataset.target_std).to(device)

    fill_bins = dataset.fill_bins[indices]
    results = {}

    for bin_idx, bin_name in enumerate(FILL_BIN_NAMES):
        bin_mask = fill_bins == bin_idx
        bin_indices = indices[bin_mask]
        if len(bin_indices) == 0:
            results[bin_name] = {"mse": float("nan"), "mag_err_db": float("nan"), "n": 0}
            continue

        subset = Subset(dataset, bin_indices.tolist())
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

        total_mse, total_mag, n = 0.0, 0.0, 0
        for batch in loader:
            grids, targets = batch[0].to(device), batch[1].to(device)
            pred = model(grids)
            total_mse += nn.functional.mse_loss(pred, targets).item() * grids.shape[0]

            pred_raw = pred * target_std + target_mean
            tgt_raw = targets * target_std + target_mean
            eps = 1e-6
            p_db = 20 * torch.log10(torch.sqrt(pred_raw[..., 0]**2 + pred_raw[..., 1]**2 + eps))
            t_db = 20 * torch.log10(torch.sqrt(tgt_raw[..., 0]**2 + tgt_raw[..., 1]**2 + eps))
            total_mag += (p_db - t_db).abs().mean().item() * grids.shape[0]
            n += grids.shape[0]

        results[bin_name] = {
            "mse": total_mse / n,
            "mag_err_db": total_mag / n,
            "n": n,
        }

    return results


def make_balanced_train_loader(
    dataset: PixelGridDataset,
    train_indices: np.ndarray,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader with fill-balanced sampling for the training split."""
    # Compute per-sample weights based on fill bin for the training subset
    fill_bins = dataset.fill_bins[train_indices]
    bin_counts = np.bincount(fill_bins, minlength=len(FILL_BIN_NAMES))
    bin_weights = 1.0 / np.maximum(bin_counts, 1).astype(np.float64)
    sample_weights = bin_weights[fill_bins]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True,
    )

    train_subset = Subset(dataset, train_indices.tolist())
    return DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train S-parameter surrogate CNN")
    parser.add_argument("--dataset-dir", type=str, nargs="+",
                        default=[os.environ.get("INVDESIGN_DATASET",
                                                "./datasets/pixelgrid")])
    parser.add_argument("--max-designs", type=int, default=None,
                        help="Limit number of designs to load (for quick testing)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-balanced", action="store_true",
                        help="Disable fill-balanced sampling (use uniform random)")
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2", "v3", "v4"],
                        help="Model architecture: v1=original CNN, v2=multi-scale, v3=ResNet+cross-attention")
    parser.add_argument("--finetune-epochs", type=int, default=0,
                        help="Extra epochs fine-tuning on well-coupled subset only (0=disabled)")
    parser.add_argument("--finetune-lr", type=float, default=1e-4,
                        help="Learning rate for fine-tuning phase")
    parser.add_argument("--save-dir", type=str, default="surrogate/checkpoints")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for train/val split and weight init (vary for ensemble)")
    parser.add_argument("--early-stop-patience", type=int, default=30,
                        help="Stop training if val_mse does not improve for this many epochs. "
                             "0 disables early stopping (Karahan et al. 2024 technique).")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4,
                        help="Minimum val_mse improvement to reset the patience counter.")
    parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae", "huber"],
                        help="Reconstruction loss on normalized real/imag. Karahan et al. use MAE.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="FC head dropout (v4). Karahan et al. use 0.5; prior default was 0.15.")
    parser.add_argument("--output-activation", type=str, default="none", choices=["none", "tanh"],
                        help="v4 output activation for Stage 1. 'none' for in-air pretrain (avoids "
                             "vanishing gradients per Karahan JSSC 2023); 'tanh' for dielectric.")
    parser.add_argument("--finetune-output-activation", type=str, default="tanh",
                        choices=["none", "tanh", "same"],
                        help="v4 output activation for Stage 2 fine-tune. 'same' inherits Stage 1; "
                             "'tanh' completes the Karahan delayed-activation transfer protocol.")
    parser.add_argument("--w-mse", type=float, default=1.0,
                        help="Weight on the (MSE/MAE/Huber) reconstruction term.")
    parser.add_argument("--w-logmag", type=float, default=0.5,
                        help="Weight on the log-magnitude L2 term. Raise to push gradient toward "
                             "relative-error regions (sparse bin) at the cost of absolute error.")
    parser.add_argument("--w-phase", type=float, default=0.1,
                        help="Weight on the phase cosine-distance term.")
    parser.add_argument("--fill-min", type=float, default=0.0,
                        help="Drop designs with inner fill fraction below this value. "
                             "Use e.g. 0.15 to remove the extreme-sparse tail.")
    parser.add_argument("--fill-max", type=float, default=1.0,
                        help="Drop designs with inner fill fraction above this value. "
                             "Use e.g. 0.85 to remove the extreme-dense tail.")
    parser.add_argument("--disconnect-aug-prob", type=float, default=0.0,
                        help="Probability of replacing a training sample with a "
                             "synthetic (disconnected speckle grid, ~-80 dB coupling) "
                             "pair. Cures the 'hallucinate -38 dB on empty grids' "
                             "failure mode. Try 0.25-0.4 for GA-facing checkpoints.")
    parser.add_argument("--disconnect-aug-db", type=float, default=-80.0,
                        help="Target |S_ij| in dB for disconnected-grid augmentation.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print("Loading dataset...")
    t0 = time.time()
    dataset = PixelGridDataset(
        args.dataset_dir,
        max_designs=args.max_designs,
        augment=not args.no_augment,
        fill_min=args.fill_min,
        fill_max=args.fill_max,
        disconnect_aug_prob=args.disconnect_aug_prob,
        disconnect_aug_db=args.disconnect_aug_db,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Train/val split (index-based to preserve fill_bins tracking)
    n_total = len(dataset)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=rng).numpy()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    # Report fill distribution in train/val
    for split_name, split_idx in [("Train", train_indices), ("Val", val_indices)]:
        bins = dataset.fill_bins[split_idx]
        counts = np.bincount(bins, minlength=len(FILL_BIN_NAMES))
        dist_str = ", ".join(f"{FILL_BIN_NAMES[i]}={counts[i]}" for i in range(len(FILL_BIN_NAMES)))
        print(f"  {split_name}: {len(split_idx)} samples — {dist_str}")

    # Create data loaders
    if not args.no_balanced:
        print("  Using fill-balanced sampling for training")
        train_loader = make_balanced_train_loader(
            dataset, train_indices, args.batch_size,
        )
    else:
        train_subset = Subset(dataset, train_indices.tolist())
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)

    val_subset = Subset(dataset, val_indices.tolist())
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    if args.model_version == "v4":
        model = SParamEM300(
            base_channels=args.base_channels,
            dropout=args.dropout,
            output_activation=args.output_activation,
        ).to(device)
    elif args.model_version == "v3":
        model = SParamResNet(base_channels=args.base_channels).to(device)
    elif args.model_version == "v2":
        model = SParamCNNv2(base_channels=args.base_channels).to(device)
    else:
        model = SParamCNN(base_channels=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {args.model_version} — {n_params:,} parameters")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_mse = float("inf")
    epochs_since_improvement = 0
    history: list[dict] = []

    print(f"\nTraining for {args.epochs} epochs (early-stop patience={args.early_stop_patience})...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device, dataset,
                                    w_mse=args.w_mse, w_logmag=args.w_logmag, w_phase=args.w_phase,
                                    loss_kind=args.loss)
        val_metrics = eval_epoch(model, val_loader, device, dataset)
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "lr": lr,
            "epoch_time_s": dt,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_logmag": train_metrics["logmag"],
            "train_phase": train_metrics["phase"],
            "val_mse": val_metrics["mse"],
            "val_logmag": val_metrics["logmag"],
            "val_mag_err_db": val_metrics["mag_err_db"],
        }

        # Stratified eval every 25 epochs + last epoch
        if epoch % 25 == 0 or epoch == args.epochs:
            strat = eval_stratified(model, dataset, val_indices, device)
            strat_record = {}
            for bin_name, metrics in strat.items():
                safe_name = bin_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct").replace("<", "lt").replace(">", "gt")
                strat_record[f"val_{safe_name}_mse"] = metrics["mse"]
                strat_record[f"val_{safe_name}_mag_err_db"] = metrics["mag_err_db"]
                strat_record[f"val_{safe_name}_n"] = metrics["n"]
            record.update(strat_record)

        history.append(record)

        improved = val_metrics["mse"] < best_val_mse - args.early_stop_min_delta
        if improved:
            best_val_mse = val_metrics["mse"]
            epochs_since_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mse": best_val_mse,
                "target_mean": dataset.target_mean,
                "target_std": dataset.target_std,
                "args": vars(args),
            }, save_dir / "best_model.pt")
        else:
            epochs_since_improvement += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            flag = " *" if improved else ""
            print(
                f"Epoch {epoch:3d}/{args.epochs} ({dt:.1f}s) | "
                f"train_loss={train_metrics['loss']:.4f} mse={train_metrics['mse']:.4f} | "
                f"val_mse={val_metrics['mse']:.4f} logmag={val_metrics['logmag']:.4f} "
                f"mag_err={val_metrics['mag_err_db']:.1f}dB | lr={lr:.1e}{flag}"
            )

        # Print stratified results when computed
        if epoch % 25 == 0 or epoch == args.epochs:
            print(f"  Stratified val (epoch {epoch}):")
            for bin_name, metrics in strat.items():
                if metrics["n"] > 0:
                    print(f"    {bin_name}: MSE={metrics['mse']:.4f} "
                          f"mag_err={metrics['mag_err_db']:.1f}dB (n={metrics['n']})")

        if args.early_stop_patience > 0 and epochs_since_improvement >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}: no val_mse improvement for "
                  f"{args.early_stop_patience} epochs (best val_mse={best_val_mse:.6f})")
            break

    # Save training history
    with (save_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    # ── Fine-tuning on well-coupled subset ──────────────────────────────
    if args.finetune_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Fine-tuning on well-coupled samples for {args.finetune_epochs} epochs")
        print(f"{'='*60}")

        # Load best model from main training
        ckpt = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        # Stage 2: optionally swap the output activation in-place (Karahan JSSC
        # 2023 delayed-Tanh transfer protocol). The Linear head weights are
        # preserved; only the trailing nonlinearity is added or removed.
        ft_act = args.finetune_output_activation
        if ft_act != "same" and ft_act != args.output_activation:
            print(f"  Stage 2: swapping output activation "
                  f"{args.output_activation!r} → {ft_act!r}")
            model.set_output_activation(ft_act)

        # Filter training indices to well-coupled samples (coupling_weight > 1)
        ft_mask = dataset.coupling_weights[train_indices] > 1.0
        ft_train_indices = train_indices[ft_mask]
        ft_val_mask = dataset.coupling_weights[val_indices] > 1.0
        ft_val_indices = val_indices[ft_val_mask]
        print(f"  Fine-tune train: {len(ft_train_indices)} / {len(train_indices)} samples")
        print(f"  Fine-tune val:   {len(ft_val_indices)} / {len(val_indices)} samples")

        ft_train_loader = make_balanced_train_loader(
            dataset, ft_train_indices, args.batch_size,
        )
        ft_val_subset = Subset(dataset, ft_val_indices.tolist())
        ft_val_loader = DataLoader(ft_val_subset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=4, pin_memory=True)

        ft_optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=1e-4)
        ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, T_max=args.finetune_epochs)

        best_ft_val = float("inf")
        for epoch in range(1, args.finetune_epochs + 1):
            t0 = time.time()
            train_metrics = train_epoch(model, ft_train_loader, ft_optimizer, device, dataset,
                                        w_mse=args.w_mse, w_logmag=args.w_logmag, w_phase=args.w_phase,
                                        loss_kind=args.loss)
            val_metrics = eval_epoch(model, ft_val_loader, device, dataset)
            ft_scheduler.step()
            dt = time.time() - t0

            improved = val_metrics["mse"] < best_ft_val
            if improved:
                best_ft_val = val_metrics["mse"]
                torch.save({
                    "epoch": args.epochs + epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": ft_optimizer.state_dict(),
                    "val_mse": best_ft_val,
                    "target_mean": dataset.target_mean,
                    "target_std": dataset.target_std,
                    "args": {**vars(args), "output_activation": model.output_activation},
                    "finetune": True,
                }, save_dir / "best_model.pt")

            if epoch % 10 == 0 or epoch == 1 or epoch == args.finetune_epochs:
                flag = " *" if improved else ""
                print(
                    f"FT {epoch:3d}/{args.finetune_epochs} ({dt:.1f}s) | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_mse={val_metrics['mse']:.4f} "
                    f"mag_err={val_metrics['mag_err_db']:.1f}dB{flag}"
                )

            if epoch == args.finetune_epochs:
                strat = eval_stratified(model, dataset, val_indices, device)
                print(f"  Stratified val (after fine-tune):")
                for bin_name, metrics in strat.items():
                    if metrics["n"] > 0:
                        print(f"    {bin_name}: MSE={metrics['mse']:.4f} "
                              f"mag_err={metrics['mag_err_db']:.1f}dB (n={metrics['n']})")

    # Save final model. Record the *current* output activation so downstream
    # inference rebuilds the model with the Tanh applied if Stage 2 swapped it.
    torch.save({
        "epoch": args.epochs + args.finetune_epochs,
        "model_state_dict": model.state_dict(),
        "target_mean": dataset.target_mean,
        "target_std": dataset.target_std,
        "args": {**vars(args), "output_activation": model.output_activation},
    }, save_dir / "final_model.pt")

    # Final stratified summary
    print(f"\nDone. Best val MSE: {best_val_mse:.6f}")
    print(f"Checkpoints saved to {save_dir}/")


if __name__ == "__main__":
    main()
