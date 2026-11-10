import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from config import Config
from dataset import EndoscopyMILDataset
from models import build_model as build_acmil_model, rdga_loss
from weights import ensure_pretrained_weight, load_pretrained_weight
from utils import (
    set_seed,
    seed_worker,
    compute_binary_metrics,
    format_metrics,
)


class Logger:
    def __init__(self, log_path):
        self.terminal_stdout = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal_stdout.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal_stdout.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def build_transforms(img_size):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform


def build_dataloaders(cfg):
    train_transform, val_transform = build_transforms(cfg.IMG_SIZE)

    train_full_dataset = EndoscopyMILDataset(
        negative_dir=cfg.TRAIN_NEGATIVE,
        positive_dir=cfg.TRAIN_POSITIVE,
        transform=train_transform,
        instances_per_bag=cfg.INSTANCES_PER_BAG,
    )

    val_full_dataset = EndoscopyMILDataset(
        negative_dir=cfg.TRAIN_NEGATIVE,
        positive_dir=cfg.TRAIN_POSITIVE,
        transform=val_transform,
        instances_per_bag=cfg.INSTANCES_PER_BAG,
    )

    if len(train_full_dataset) != len(val_full_dataset):
        raise RuntimeError(
            "Train and validation dataset lengths do not match."
        )

    num_samples = len(train_full_dataset)

    if num_samples < 2:
        raise RuntimeError(
            "At least 2 bags are required for train/validation split."
        )

    generator = torch.Generator()
    generator.manual_seed(cfg.VSEED)

    indices = torch.randperm(
        num_samples,
        generator=generator,
    ).tolist()

    num_val = int(round(num_samples * cfg.VAL_RATIO))
    num_val = max(1, min(num_val, num_samples - 1))

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    train_dataset = Subset(
        train_full_dataset,
        train_indices,
    )

    val_dataset = Subset(
        val_full_dataset,
        val_indices,
    )

    print(f"Total bags: {num_samples}")
    print(f"Train bags: {len(train_dataset)}")
    print(f"Val bags: {len(val_dataset)}")

    train_generator = torch.Generator()
    train_generator.manual_seed(cfg.SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg.NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg.NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader


def create_model(cfg, device):
    model = build_acmil_model(cfg)

    use_pretrained = (
        cfg.REQUIRE_PRETRAINED
        or bool(cfg.PRETRAINED_PATH)
        or bool(cfg.PRETRAINED_URL)
    )

    if use_pretrained:
        pretrained_path = ensure_pretrained_weight(
            pretrained_path=cfg.PRETRAINED_PATH,
            pretrained_url=cfg.PRETRAINED_URL,
            require_pretrained=cfg.REQUIRE_PRETRAINED,
        )

        if pretrained_path:
            print(
                f"Using pretrained weights: "
                f"{pretrained_path}"
            )

            if hasattr(model.encoder, "model"):
                load_pretrained_weight(
                    model.encoder.model,
                    pretrained_path,
                )
            else:
                load_pretrained_weight(
                    model.encoder,
                    pretrained_path,
                )
        else:
            print(
                "No pretrained weights found. "
                "Using randomly initialized backbone."
            )

    model = model.to(device)

    print("=" * 100)
    print("Model: RelationalDynamicGroupACMIL")
    print(f"Backbone: {cfg.ARCH}")
    print(f"Feature dim: {model.feature_dim}")
    print(f"Inner dim: {model.inner_dim}")
    print(f"Attention dim: {cfg.ATTN_DIM}")
    print(f"Number of tokens: {model.n_token}")
    print(f"Relation top-k: {model.relation_topk}")
    print(f"Masked patches: {model.n_masked_patch}")
    print(f"Mask drop: {model.mask_drop}")
    print(
        f"Group prior strength init: "
        f"{cfg.GROUP_PRIOR_STRENGTH}"
    )
    print("=" * 100)

    return model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp,
):
    model.train()

    total_loss = 0.0
    total_slide_loss = 0.0
    total_sub_loss = 0.0
    total_diff_loss = 0.0
    total_compact_loss = 0.0
    total_balance_loss = 0.0
    total_separation_loss = 0.0
    total_samples = 0

    label_chunks = []
    prob_chunks = []
    pred_chunks = []

    pbar = tqdm(
        loader,
        desc="Train",
        ncols=140,
    )

    for bags, labels, _ in pbar:
        bags = bags.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):
            logits, aux = model(
                bags,
                return_aux=True,
            )

            loss, loss_dict = rdga_loss(
                logits=logits,
                aux=aux,
                labels=labels,
                criterion=criterion,
            )

        if use_amp:
            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)

        total_samples += batch_size

        total_loss += (
            loss_dict["loss"].item()
            * batch_size
        )

        total_slide_loss += (
            loss_dict["slide_loss"].item()
            * batch_size
        )

        total_sub_loss += (
            loss_dict["sub_loss"].item()
            * batch_size
        )

        total_diff_loss += (
            loss_dict["diff_loss"].item()
            * batch_size
        )

        total_compact_loss += (
            loss_dict["group_compact_loss"].item()
            * batch_size
        )

        total_balance_loss += (
            loss_dict["group_balance_loss"].item()
            * batch_size
        )

        total_separation_loss += (
            loss_dict["group_separation_loss"].item()
            * batch_size
        )

        with torch.no_grad():
            probs = torch.softmax(
                logits.detach(),
                dim=1,
            )[:, 1]

            preds = torch.argmax(
                logits.detach(),
                dim=1,
            )

            label_chunks.append(
                labels.detach().cpu()
            )

            prob_chunks.append(
                probs.float().cpu()
            )

            pred_chunks.append(
                preds.cpu()
            )

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "slide": f"{loss_dict['slide_loss'].item():.4f}",
            "sub": f"{loss_dict['sub_loss'].item():.4f}",
            "diff": f"{loss_dict['diff_loss'].item():.4f}",
        })

    all_labels = torch.cat(
        label_chunks
    ).numpy()

    all_probs = torch.cat(
        prob_chunks
    ).numpy()

    all_preds = torch.cat(
        pred_chunks
    ).numpy()

    metrics = compute_binary_metrics(
        y_true=all_labels,
        y_prob_pos=all_probs,
        y_pred=all_preds,
    )

    loss_dict = {
        "loss": total_loss / total_samples,
        "slide_loss": (
            total_slide_loss
            / total_samples
        ),
        "sub_loss": (
            total_sub_loss
            / total_samples
        ),
        "diff_loss": (
            total_diff_loss
            / total_samples
        ),
        "group_compact_loss": (
            total_compact_loss
            / total_samples
        ),
        "group_balance_loss": (
            total_balance_loss
            / total_samples
        ),
        "group_separation_loss": (
            total_separation_loss
            / total_samples
        ),
    }

    return loss_dict, metrics


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    use_amp,
):
    model.eval()

    total_loss = 0.0
    total_slide_loss = 0.0
    total_sub_loss = 0.0
    total_diff_loss = 0.0
    total_compact_loss = 0.0
    total_balance_loss = 0.0
    total_separation_loss = 0.0
    total_samples = 0

    label_chunks = []
    prob_chunks = []
    pred_chunks = []

    pbar = tqdm(
        loader,
        desc="Val",
        ncols=140,
    )

    for bags, labels, _ in pbar:
        bags = bags.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):
            logits, aux = model(
                bags,
                return_aux=True,
            )

            loss, loss_dict = rdga_loss(
                logits=logits,
                aux=aux,
                labels=labels,
                criterion=criterion,
            )

        batch_size = labels.size(0)

        total_samples += batch_size

        total_loss += (
            loss_dict["loss"].item()
            * batch_size
        )

        total_slide_loss += (
            loss_dict["slide_loss"].item()
            * batch_size
        )

        total_sub_loss += (
            loss_dict["sub_loss"].item()
            * batch_size
        )

        total_diff_loss += (
            loss_dict["diff_loss"].item()
            * batch_size
        )

        total_compact_loss += (
            loss_dict["group_compact_loss"].item()
            * batch_size
        )

        total_balance_loss += (
            loss_dict["group_balance_loss"].item()
            * batch_size
        )

        total_separation_loss += (
            loss_dict["group_separation_loss"].item()
            * batch_size
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        preds = torch.argmax(
            logits,
            dim=1,
        )

        label_chunks.append(
            labels.cpu()
        )

        prob_chunks.append(
            probs.float().cpu()
        )

        pred_chunks.append(
            preds.cpu()
        )

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    all_labels = torch.cat(
        label_chunks
    ).numpy()

    all_probs = torch.cat(
        prob_chunks
    ).numpy()

    all_preds = torch.cat(
        pred_chunks
    ).numpy()

    metrics = compute_binary_metrics(
        y_true=all_labels,
        y_prob_pos=all_probs,
        y_pred=all_preds,
    )

    loss_dict = {
        "loss": total_loss / total_samples,
        "slide_loss": (
            total_slide_loss
            / total_samples
        ),
        "sub_loss": (
            total_sub_loss
            / total_samples
        ),
        "diff_loss": (
            total_diff_loss
            / total_samples
        ),
        "group_compact_loss": (
            total_compact_loss
            / total_samples
        ),
        "group_balance_loss": (
            total_balance_loss
            / total_samples
        ),
        "group_separation_loss": (
            total_separation_loss
            / total_samples
        ),
    }

    return loss_dict, metrics


def main():
    cfg = Config

    cfg.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    time_str = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = (
        cfg.OUTPUT_DIR
        / f"train_log_{time_str}.txt"
    )

    logger = Logger(log_path)
    sys.stdout = logger

    try:
        print(
            f"Log file: {log_path}"
        )

        set_seed(
            cfg.SEED,
            deterministic=True,
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print("=" * 100)
        print(f"Device: {device}")
        print(f"Backbone: {cfg.ARCH}")
        print(
            f"Batch size: "
            f"{cfg.BATCH_SIZE}"
        )
        print(
            f"Instances per bag: "
            f"{cfg.INSTANCES_PER_BAG}"
        )
        print(
            f"Validation ratio: "
            f"{cfg.VAL_RATIO}"
        )
        print(
            f"Epochs: "
            f"{cfg.EPOCHS}"
        )
        print(
            f"Learning rate: "
            f"{cfg.LR}"
        )
        print(
            f"Weight decay: "
            f"{cfg.WEIGHT_DECAY}"
        )
        print("=" * 100)

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            print(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

            print(
                f"GPU memory: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB"
            )

        train_loader, val_loader = (
            build_dataloaders(cfg)
        )

        model = create_model(
            cfg,
            device,
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.EPOCHS,
                eta_min=1e-6,
            )
        )

        use_amp = (
            device.type == "cuda"
            and cfg.USE_AMP
        )

        scaler = torch.cuda.amp.GradScaler(
            enabled=use_amp
        )

        print(
            f"AMP enabled: "
            f"{use_amp}"
        )

        best_auc = -1.0
        best_epoch = -1

        for epoch in range(
            1,
            cfg.EPOCHS + 1,
        ):
            print("=" * 100)

            print(
                f"Epoch "
                f"[{epoch:03d}/{cfg.EPOCHS}]"
            )

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            print(
                f"Learning rate: "
                f"{current_lr:.8f}"
            )

            train_loss_dict, train_metrics = (
                train_one_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    scaler=scaler,
                    use_amp=use_amp,
                )
            )

            val_loss_dict, val_metrics = (
                validate_one_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    use_amp=use_amp,
                )
            )

            scheduler.step()

            val_auc = val_metrics["auc"]

            improved = False

            if (
                    not np.isnan(val_auc)
                    and val_auc > best_auc
            ):
                best_auc = val_auc
                best_epoch = epoch
                improved = True

                save_path = cfg.save_path()

                torch.save(
                    {
                        "epoch": epoch,
                        "arch": cfg.ARCH,
                        "model_name": "RelationalDynamicGroupACMIL",
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_auc": best_auc,
                        "config": {
                            "instances_per_bag": cfg.INSTANCES_PER_BAG,
                            "batch_size": cfg.BATCH_SIZE,
                            "img_size": cfg.IMG_SIZE,
                            "lr": cfg.LR,
                            "weight_decay": cfg.WEIGHT_DECAY,
                            "arch": cfg.ARCH,
                            "inner_dim": cfg.INNER_DIM,
                            "attn_dim": cfg.ATTN_DIM,
                            "group_dim": cfg.GROUP_DIM,
                            "fusion_attn_dim": cfg.FUSION_ATTN_DIM,
                            "relation_dim": cfg.RELATION_DIM,
                            "relation_topk": cfg.RELATION_TOPK,
                            "relation_scale": cfg.RELATION_SCALE,
                            "n_token": cfg.N_TOKEN,
                            "n_masked_patch": cfg.N_MASKED_PATCH,
                            "mask_drop": cfg.MASK_DROP,
                            "group_temperature": cfg.GROUP_TEMPERATURE,
                            "group_prior_strength": cfg.GROUP_PRIOR_STRENGTH,
                        },
                    },
                    save_path,
                )

            print(
                f"Train Loss="
                f"{train_loss_dict['loss']:.4f} | "
                f"Slide="
                f"{train_loss_dict['slide_loss']:.4f} | "
                f"Sub="
                f"{train_loss_dict['sub_loss']:.4f} | "
                f"Diff="
                f"{train_loss_dict['diff_loss']:.4f}"
            )

            print(
                f"Train Group | "
                f"Compact="
                f"{train_loss_dict['group_compact_loss']:.4f} | "
                f"Balance="
                f"{train_loss_dict['group_balance_loss']:.4f} | "
                f"Separation="
                f"{train_loss_dict['group_separation_loss']:.4f}"
            )

            print(
                f"Train Metrics | "
                f"{format_metrics(train_metrics)}"
            )

            print(
                f"Val   Loss="
                f"{val_loss_dict['loss']:.4f} | "
                f"Slide="
                f"{val_loss_dict['slide_loss']:.4f} | "
                f"Sub="
                f"{val_loss_dict['sub_loss']:.4f} | "
                f"Diff="
                f"{val_loss_dict['diff_loss']:.4f}"
            )

            print(
                f"Val Group | "
                f"Compact="
                f"{val_loss_dict['group_compact_loss']:.4f} | "
                f"Balance="
                f"{val_loss_dict['group_balance_loss']:.4f} | "
                f"Separation="
                f"{val_loss_dict['group_separation_loss']:.4f}"
            )

            print(
                f"Val Metrics | "
                f"{format_metrics(val_metrics)}"
            )

            print(
                f"Best Val AUC="
                f"{best_auc:.4f} "
                f"at epoch {best_epoch}"
                f"{' <-- saved' if improved else ''}"
            )

        print("=" * 100)
        print("Training completed.")

        print(
            f"Best validation AUC: "
            f"{best_auc:.4f}, "
            f"epoch={best_epoch}"
        )

        print(
            f"Best model saved to: "
            f"{cfg.save_path()}"
        )

        print(
            f"Log file saved to: "
            f"{log_path}"
        )

    finally:
        sys.stdout = (
            logger.terminal_stdout
        )
        logger.close()


if __name__ == "__main__":
    main()
