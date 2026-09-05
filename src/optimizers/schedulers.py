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

