"""
Trainer -- the training loop shared by every model in this project.

Originally this loop was copy-pasted, with small variations, into 9
separate scripts:
  train_resnet.py, train_resnet_100K.py, train_resnet_1M.py, train_resnet_6M.py,
  train_densenet_1M.py, train_vit_1M.py,
  train_gate_model.py, train_specialist_mel_bkl.py
This class is a single, configurable version of that loop -- see
src/train/config.py (TrainConfig) for every knob, and configs/train/*.json
for presets that reproduce each original script's exact defaults.

Usage:
    from src.models.config import ModelConfig
    from src.train.config import TrainConfig
    from src.train.trainer import Trainer

    model_cfg = ModelConfig(backbone="densenet121", num_classes=2)
    train_cfg = TrainConfig.from_json("configs/train/gate_1m.json")
    Trainer(model_cfg, train_cfg).run()
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from ..data.dataset import AugmentedSkinDataset
from ..data.splits import load_meta_and_split, load_gate_split, load_filtered_split
from ..models.losses import FocalLoss
from ..utils.gpu import (pick_device, unwrap_model, maybe_data_parallel, print_gpu_info,
                          set_seed, seed_worker, find_max_batch_size)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transforms(augmentation_level):
    if augmentation_level == "basic":
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # "full"
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def _load_split(cfg):
    if cfg.split_mode == "gate":
        return load_gate_split(
            cfg.meta_csv, cfg.val_fraction, cfg.test_fraction, cfg.seed,
            cancer_classes=cfg.cancer_classes,
            metadata_csv=cfg.metadata_csv, id_col=cfg.id_col, label_col=cfg.label_col,
        )
    if cfg.split_mode == "filtered":
        if not cfg.target_classes:
            raise ValueError("split_mode='filtered' requires TrainConfig.target_classes")
        return load_filtered_split(
            cfg.meta_csv, cfg.val_fraction, cfg.test_fraction, cfg.seed, cfg.target_classes,
            metadata_csv=cfg.metadata_csv, id_col=cfg.id_col, label_col=cfg.label_col,
        )
    return load_meta_and_split(
        cfg.meta_csv, cfg.val_fraction, cfg.test_fraction, cfg.seed,
        metadata_csv=cfg.metadata_csv, id_col=cfg.id_col, label_col=cfg.label_col,
    )


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, best_val_metric,
                     counter, class_to_idx, scaler=None):
    torch.save({
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_val_metric": best_val_metric,
        "counter": counter,
        "class_to_idx": class_to_idx,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    best_val_metric = ckpt.get("best_val_metric", 0.0)
    return ckpt["epoch"], ckpt["step"], best_val_metric, ckpt["counter"], ckpt["class_to_idx"]


class Trainer:
    def __init__(self, model_config, train_config):
        self.model_config = model_config
        self.cfg = train_config

    def run(self):
        cfg = self.cfg
        set_seed(cfg.seed)

        # --- 1. data --------------------------------------------------------
        train_tf, eval_tf = _build_transforms(cfg.augmentation_level)
        train_rows, val_rows, test_rows, class_to_idx = _load_split(cfg)
        num_classes = len(class_to_idx)
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        if num_classes != self.model_config.num_classes:
            raise ValueError(
                f"ModelConfig.num_classes={self.model_config.num_classes} but the "
                f"split produced {num_classes} classes ({list(class_to_idx)}) -- "
                f"these must match."
            )

        class_counts = np.bincount([label for _, label in train_rows], minlength=num_classes)
        raw_weights = class_counts.sum() / (num_classes * np.maximum(class_counts, 1))
        loss_weights = raw_weights ** cfg.class_weight_power
        sampler_weights = raw_weights ** cfg.sampler_weight_power
        for cls_name, mult in (cfg.extra_class_weight or {}).items():
            idx = class_to_idx[cls_name]
            loss_weights[idx] *= mult
            sampler_weights[idx] *= mult
        print(f"[CLASS BALANCE] train-set counts per class "
              f"(loss_power={cfg.class_weight_power}, sampler_power={cfg.sampler_weight_power}, "
              f"extra_weight={cfg.extra_class_weight}):")
        for i in range(num_classes):
            print(f"    {idx_to_class[i]:>10}: {class_counts[i]:>8,} rows  "
                  f"(loss weight: {loss_weights[i]:.3f}, sampler weight: {sampler_weights[i]:.3f})")

        dataset_train = AugmentedSkinDataset(train_rows, transform=train_tf)
        dataset_val = AugmentedSkinDataset(val_rows, transform=eval_tf)
        dataset_test = AugmentedSkinDataset(test_rows, transform=eval_tf)

        # --- 2. device / model ------------------------------------------------
        print_gpu_info()
        device, gpu_ids = pick_device(cfg.gpu_ids)
        pin_memory = device.type == "cuda"
        use_amp = device.type == "cuda" and cfg.use_amp
        print(f"[AMP] Mixed precision training: {'ON' if use_amp else 'OFF'}")

        model = self.model_config.build()
        model = model.to(device, memory_format=torch.channels_last) if device.type == "cuda" else model.to(device)
        if cfg.multi_gpu:
            model = maybe_data_parallel(model, gpu_ids)

        # --- 3. loss / optimizer / scheduler -----------------------------------
        alpha = torch.tensor(loss_weights, dtype=torch.float32).to(device) if cfg.class_weighted_loss else None
        if cfg.loss_type == "focal":
            criterion = FocalLoss(alpha=alpha, gamma=cfg.focal_gamma,
                                   label_smoothing=cfg.label_smoothing).to(device)
            print(f"[LOSS] Focal loss (gamma={cfg.focal_gamma}), class weighting: "
                  f"{'ON' if alpha is not None else 'OFF'}")
        else:
            criterion = nn.CrossEntropyLoss(weight=alpha, label_smoothing=cfg.label_smoothing)
            print(f"[LOSS] CrossEntropy, class weighting: {'ON' if alpha is not None else 'OFF'}")

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        if cfg.optimizer_type == "adamw":
            optimizer = optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            optimizer = optim.Adam(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        scheduler = None
        if cfg.lr_schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=2)
            print("[LR SCHEDULE] ReduceLROnPlateau on priority recall (factor=0.1, patience=2 epochs)")
        elif cfg.lr_schedule == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=0.1)
            print(f"[LR SCHEDULE] StepLR every {cfg.lr_step_size} epochs (gamma=0.1)")
        else:
            print("[LR SCHEDULE] none (fixed learning rate)")

        if cfg.auto_batch_size:
            cfg.batch_size = find_max_batch_size(
                model, optimizer, criterion, device, num_classes,
                start_batch=cfg.batch_size, max_batch=cfg.probe_max_batch, use_amp=use_amp,
            )
            print(f"[AUTOTUNE] Training will use batch_size={cfg.batch_size}")

        # --- 4. data loaders --------------------------------------------------
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(cfg.seed)

        if cfg.oversample_minority:
            per_row_weights = np.array([sampler_weights[label] for _, label in train_rows], dtype=np.float64)
            train_sampler = WeightedRandomSampler(
                weights=per_row_weights, num_samples=len(train_rows),
                replacement=True, generator=dataloader_generator,
            )
            print("[CLASS BALANCE] Minority oversampling: ON (WeightedRandomSampler)")
        else:
            train_sampler = None
            print("[CLASS BALANCE] Minority oversampling: OFF")

        train_loader = DataLoader(
            dataset_train, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, num_workers=cfg.workers, pin_memory=pin_memory,
            persistent_workers=cfg.workers > 0, worker_init_fn=seed_worker,
            generator=dataloader_generator,
        )
        val_loader = DataLoader(
            dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers,
            pin_memory=pin_memory, persistent_workers=cfg.workers > 0, worker_init_fn=seed_worker,
        )
        test_loader = DataLoader(
            dataset_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers,
            pin_memory=pin_memory, persistent_workers=cfg.workers > 0, worker_init_fn=seed_worker,
        )

        # --- 5. resume ----------------------------------------------------------
        start_epoch, resume_step, best_val_metric, counter = 0, 0, 0.0, 0
        scaler = GradScaler("cuda", enabled=use_amp)
        if cfg.resume and os.path.exists(cfg.checkpoint_path):
            start_epoch, resume_step, best_val_metric, counter, saved_class_to_idx = load_checkpoint(
                cfg.checkpoint_path, model, optimizer, scheduler, scaler=scaler
            )
            if saved_class_to_idx != class_to_idx:
                print("[WARN] class_to_idx from checkpoint differs from current data. "
                      "Proceeding with current data's class_to_idx.")
            print(f"[RESUME] epoch {start_epoch}, step {resume_step}, "
                  f"best={best_val_metric:.4f}, patience_counter={counter}")

        priority_idxs = [class_to_idx[c] for c in (cfg.priority_classes or []) if c in class_to_idx]

        # --- 6. training loop -----------------------------------------------------
        print("Starting training...")
        start_time = time.time()

        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            skip_until = resume_step if epoch == start_epoch else 0
            num_steps = len(train_loader)
            epoch_start_time = time.time()
            print(f"[EPOCH {epoch+1:02d}/{cfg.epochs}] starting -- {num_steps:,} steps "
                  f"(batch_size={cfg.batch_size})", flush=True)

            for step, (inputs, labels) in enumerate(train_loader):
                if step < skip_until:
                    continue
                if device.type == "cuda":
                    inputs = inputs.to(device, memory_format=torch.channels_last)
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with autocast("cuda", enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                steps_done = step + 1 - skip_until
                if steps_done > 0 and (steps_done % cfg.log_every == 0 or step + 1 == num_steps):
                    elapsed = time.time() - epoch_start_time
                    rate = steps_done / max(elapsed, 1e-6)
                    eta = (num_steps - (step + 1)) / max(rate, 1e-6)
                    print(f"[EPOCH {epoch+1:02d}] step {step+1:,}/{num_steps:,} "
                          f"({100*(step+1)/num_steps:.1f}%) | loss: {running_loss/steps_done:.4f} "
                          f"acc: {100*correct/max(1,total):.2f}% | {rate:.2f} it/s | "
                          f"elapsed: {elapsed/60:.1f}m ETA: {eta/60:.1f}m", flush=True)

                if cfg.checkpoint_every_steps and (step + 1) % cfg.checkpoint_every_steps == 0:
                    save_checkpoint(cfg.checkpoint_path, model, optimizer, scheduler,
                                     epoch, step + 1, best_val_metric, counter, class_to_idx, scaler=scaler)
                    print(f"[CHECKPOINT] saved at epoch {epoch+1}, step {step+1}", flush=True)

            train_loss = running_loss / max(1, (len(train_loader) - skip_until))
            train_acc = 100 * correct / max(1, total)
            print(f"[EPOCH {epoch+1:02d}] training done in {(time.time()-epoch_start_time)/60:.1f}m "
                  f"-- running validation...", flush=True)

            # --- validation ---
            val_acc, val_loss, val_priority_recall, val_macro_recall = self._evaluate(
                model, val_loader, criterion, device, num_classes, priority_idxs, use_amp
            )
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | Macro Recall: {val_macro_recall:.4f} | "
                  f"Priority ({','.join(cfg.priority_classes) or 'macro'}) Recall: {val_priority_recall:.4f}",
                  flush=True)

            if scheduler is not None:
                prev_lr = optimizer.param_groups[0]["lr"]
                if cfg.lr_schedule == "plateau":
                    scheduler.step(val_priority_recall)
                else:
                    scheduler.step()
                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr != prev_lr:
                    print(f"[LR SCHEDULE] LR reduced: {prev_lr:.2e} -> {new_lr:.2e}", flush=True)

            save_checkpoint(cfg.checkpoint_path, model, optimizer, scheduler,
                             epoch + 1, 0, best_val_metric, counter, class_to_idx, scaler=scaler)

            if val_priority_recall > best_val_metric:
                best_val_metric = val_priority_recall
                counter = 0
                torch.save(unwrap_model(model).state_dict(), cfg.best_model_path)
                print(f"  -> new best (val priority recall = {best_val_metric:.4f}), "
                      f"saved to {cfg.best_model_path}")
            else:
                counter += 1
                if counter >= cfg.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Training complete. Total time: {(time.time()-start_time)/60:.2f} minutes")

        # --- 7. final test evaluation --------------------------------------------
        print("Evaluating best model on test set...")
        unwrap_model(model).load_state_dict(torch.load(cfg.best_model_path, map_location=device))
        test_acc, test_loss, test_priority_recall, test_macro_recall = self._evaluate(
            model, test_loader, criterion, device, num_classes, priority_idxs, use_amp
        )
        print(f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%  "
              f"Macro Recall: {test_macro_recall:.4f}  Priority Recall: {test_priority_recall:.4f}")
        return {
            "class_to_idx": class_to_idx,
            "best_model_path": cfg.best_model_path,
            "test_accuracy": test_acc,
            "test_macro_recall": test_macro_recall,
            "test_priority_recall": test_priority_recall,
        }

    @staticmethod
    @torch.no_grad()
    def _evaluate(model, loader, criterion, device, num_classes, priority_idxs, use_amp):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        for inputs, labels in loader:
            if device.type == "cuda":
                inputs = inputs.to(device, memory_format=torch.channels_last)
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            with autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        acc = 100 * correct / max(1, total)
        avg_loss = total_loss / max(1, len(loader))
        per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0,
                                         labels=list(range(num_classes)))
        macro_recall = float(np.mean(per_class_recall))
        if priority_idxs:
            priority_recall = float(np.mean([per_class_recall[i] for i in priority_idxs]))
        else:
            priority_recall = macro_recall
        return acc, avg_loss, priority_recall, macro_recall
