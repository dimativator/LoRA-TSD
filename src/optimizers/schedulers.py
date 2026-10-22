def make_linear_warmup_decay(base_lr: float, total_steps: int, warmup_frac: float = 0.1, min_lr: float = 0.0):
    warmup_steps = max(1, int(round(warmup_frac * total_steps)))
    total_steps = max(1, int(total_steps))

    def sched(step_idx: int, _base_lr_unused: float, _group_idx: int) -> float:
        s = int(step_idx)
        if s <= 0:
            return 0.0

        if s < warmup_steps:
            lr = base_lr * (s / warmup_steps)
        else:
            denom = max(1, total_steps - warmup_steps)
            t = min(max((s - warmup_steps) / denom, 0.0), 1.0)
            lr = base_lr + (min_lr - base_lr) * t

        return max(float(lr), float(min_lr))

    return sched


def make_torch_linear_warmup_decay(
    optimizer, total_steps: int, warmup_frac: float = 0.1, min_lr: float = 0.0
):
    """Linear warmup and decay for standard torch optimizers.

    The first optimizer update uses ``base_lr / warmup_steps`` and the final
    update uses ``min_lr``.
    """
    import torch

    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(round(warmup_frac * total_steps)))

    def make_lr_lambda(base_lr):
        floor = min(float(min_lr), float(base_lr))
        floor_ratio = floor / float(base_lr) if base_lr > 0 else 0.0

        def lr_lambda(scheduler_step):
            update_step = min(int(scheduler_step) + 1, total_steps)
            if update_step <= warmup_steps:
                return update_step / warmup_steps
            decay_steps = max(1, total_steps - warmup_steps)
            progress = (update_step - warmup_steps) / decay_steps
            return 1.0 + (floor_ratio - 1.0) * progress

        return lr_lambda

    lambdas = [make_lr_lambda(group["lr"]) for group in optimizer.param_groups]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
