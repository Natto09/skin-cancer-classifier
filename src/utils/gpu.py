import copy
import random

import numpy as np
import torch
import torch.nn as nn


def pick_device(gpu_ids=None):
    """Returns (device, gpu_ids). If gpu_ids is None, uses every visible GPU."""
    if torch.cuda.is_available():
        ids = gpu_ids if gpu_ids else list(range(torch.cuda.device_count()))
        return torch.device(f"cuda:{ids[0]}"), ids
    return torch.device("cpu"), []


def unwrap_model(model):
    """Returns the underlying model whether or not it's wrapped in DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_data_parallel(model, gpu_ids):
    """Wraps in DataParallel only if there's more than one GPU to use."""
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        print(f"[GPU] Using DataParallel across GPUs: {gpu_ids}")
        return nn.DataParallel(model, device_ids=gpu_ids)
    return model


def print_gpu_info():
    if not torch.cuda.is_available():
        print("[GPU] No CUDA device available -- running on CPU.")
        return
    n = torch.cuda.device_count()
    print(f"[GPU] {n} CUDA device(s) visible:")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"       [{i}] {props.name} -- {total_gb:.1f} GB total memory")


def set_seed(seed):
    """Fixes every source of randomness we control, so reruns use the same
    train/val/test split, the same shuffle order each epoch, and the same
    initial conditions."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Gives each DataLoader worker process its own deterministic seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def find_max_batch_size(model, optimizer, criterion, device, num_classes,
                         start_batch, max_batch, use_amp=True,
                         growth_factor=2, safety_margin=0.9, refine_steps=6):
    """
    Probes how large a batch size fits in GPU memory in two passes:

    1. COARSE: doubles the batch size until it hits an out-of-memory error.
    2. REFINE: binary-searches within that gap for `refine_steps` rounds to
       narrow in on the actual boundary.

    `safety_margin` then backs off a bit further from that boundary, since
    real training adds overhead this synthetic probe doesn't see (DataLoader
    workers, pinned-memory buffers, CUDA fragmentation over many steps).

    Model and optimizer state are restored afterward so this doesn't
    corrupt the pretrained weights before real training starts.
    """
    from torch.amp import autocast, GradScaler

    if device.type != "cuda":
        print("[AUTOTUNE] Not running on CUDA -- skipping batch size probe, "
              f"using batch_size as given ({start_batch}).")
        return start_batch

    amp_tag = "with AMP" if use_amp else "without AMP"
    print(f"[AUTOTUNE] Probing max batch size that fits in GPU memory ({amp_tag}), "
          f"ceiling={max_batch:,} ...")
    model_state = copy.deepcopy(unwrap_model(model).state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scaler = GradScaler("cuda", enabled=use_amp)

    def try_batch(batch):
        try:
            torch.cuda.empty_cache()
            dummy_inputs = torch.randn(batch, 3, 224, 224, device=device)
            dummy_labels = torch.randint(0, num_classes, (batch,), device=device)
            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                outputs = model(dummy_inputs)
                loss = criterion(outputs, dummy_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            raise

    last_good = start_batch
    first_bad = None
    try:
        model.train()
        batch = start_batch
        while batch <= max_batch:
            if try_batch(batch):
                print(f"[AUTOTUNE] batch_size={batch} OK")
                last_good = batch
                if batch == max_batch:
                    break
                batch = min(batch * growth_factor, max_batch)
            else:
                print(f"[AUTOTUNE] batch_size={batch} ran out of memory -- stopping coarse search")
                first_bad = batch
                break

        if first_bad is not None:
            lo, hi = last_good, first_bad
            print(f"[AUTOTUNE] Refining between {lo} (OK) and {hi} (OOM) ...")
            for _ in range(refine_steps):
                if hi - lo <= 1:
                    break
                mid = (lo + hi) // 2
                if try_batch(mid):
                    print(f"[AUTOTUNE] batch_size={mid} OK")
                    lo = mid
                else:
                    print(f"[AUTOTUNE] batch_size={mid} ran out of memory")
                    hi = mid
            last_good = lo
    finally:
        unwrap_model(model).load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        torch.cuda.empty_cache()

    safe_batch = max(start_batch, int(last_good * safety_margin))
    print(f"[AUTOTUNE] Largest batch size that fit: {last_good}. "
          f"Using batch size with safety margin: {safe_batch}")
    return safe_batch
