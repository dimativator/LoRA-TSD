"""Atomically share untrained LOI factors across every direction and learning rate."""

import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Callable

import torch


def factor_digest(factors: list) -> str:
    digest = hashlib.sha256()
    for pair in factors:
        for tensor in pair:
            digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initial_factors(campaign: Path, config: dict, groups: list,
                    generate: Callable[[], None], guard: Callable) -> dict:
    """Generate once, publish atomically, then validate every tensor before copying.

    This stores only the initial A/B tensors. It never stores trained weights,
    optimizer states, model weights, or training checkpoints.
    """
    metadata = {key: config[key] for key in (
        "model", "model_revision", "lora_r", "rsvd_q", "rsvd_scale", "dataset", "seed")}
    metadata.update(schema=1, train_microbatch=config["train_batch_size"],
                    lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
                    target_modules=list(config["lora_target_modules"]))
    directory = campaign / "initializations"
    guard(pending_write_bytes=1_000_000, reason="before_initialization_cache_directory",
          write_path=campaign)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"init-{config['dataset']}-s{config['seed']}"
    path = directory / (stem + ".pt")
    claim = directory / (stem + ".claim.json")
    failed = directory / (stem + ".failure.json")
    owner = False
    if not path.exists():
        try:
            with claim.open("x") as stream:
                json.dump({"hostname": socket.gethostname(), "pid": os.getpid(),
                           "timestamp": time.time(), "metadata": metadata}, stream)
            owner = True
        except FileExistsError:
            pass
    if owner:
        temporary = directory / f".{stem}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            generate()
            factors = [[parameter.detach().float().cpu().contiguous().clone()
                        for parameter in group["params"]] for group in groups]
            expected_bytes = sum(tensor.numel() * tensor.element_size()
                                 for pair in factors for tensor in pair)
            guard(pending_write_bytes=expected_bytes + 1_000_000,
                  reason="before_initial_factor_cache_write", write_path=directory)
            payload = {"metadata": metadata, "factors": factors,
                       "adapter_sha256": factor_digest(factors),
                       "created_at": time.time(), "creator_pid": os.getpid(),
                       "creator_host": socket.gethostname(), "trained_checkpoint": False}
            torch.save(payload, temporary)
            # Atomic replace publishes a complete cache to concurrent readers.
            temporary.replace(path)
        except BaseException as error:
            record = {"timestamp": time.time(), "pid": os.getpid(),
                      "host": socket.gethostname(), "error": repr(error)}
            print("INITIALIZATION_CACHE_FAILED " + json.dumps(record), flush=True)
            try:
                failed.write_text(json.dumps(record) + "\n")
            except OSError:
                pass
            raise
        finally:
            if temporary.exists():
                temporary.unlink()
    while not path.exists():
        guard(reason="waiting_for_shared_initialization")
        if (campaign / "STOP_FAILURE.json").exists() or failed.exists():
            raise RuntimeError("Shared LOI initialization failed or campaign stopped")
        time.sleep(1)
    guard(reason="before_shared_initialization_read")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("metadata") != metadata or payload.get("trained_checkpoint") is not False:
        raise RuntimeError("Initial factor cache metadata mismatch")
    factors = payload.get("factors")
    if not isinstance(factors, list) or len(factors) != len(groups):
        raise RuntimeError("Initial factor cache group count mismatch")
    for group, pair in zip(groups, factors):
        if not isinstance(pair, list) or len(pair) != len(group["params"]):
            raise RuntimeError("Initial factor cache pair count mismatch")
        for parameter, tensor in zip(group["params"], pair):
            if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                    or tensor.shape != parameter.shape or not torch.isfinite(tensor).all()):
                raise RuntimeError("Invalid initial factor tensor")
    digest = factor_digest(factors)
    if digest != payload.get("adapter_sha256"):
        raise RuntimeError("Initial factor cache checksum mismatch")
    # Validate the complete payload first, so a malformed later tensor cannot
    # leave the model partially initialized.
    with torch.no_grad():
        for group, pair in zip(groups, factors):
            for parameter, tensor in zip(group["params"], pair):
                parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    record = {"path": str(path), "status": "generated" if owner else "reused",
              "adapter_sha256": digest, "metadata": metadata,
              "creator_host": payload["creator_host"], "creator_pid": payload["creator_pid"],
              "created_at": payload["created_at"], "trained_checkpoint": False}
    print("INITIALIZATION_CACHE " + json.dumps(record), flush=True)
    return record


def self_test() -> dict:
    """Verify concurrent claim uniqueness, exact reuse, and validation before copy."""
    from concurrent.futures import ThreadPoolExecutor
    import tempfile
    import threading

    config = {"model": "tiny", "model_revision": "abc", "lora_r": 2,
              "rsvd_q": 15, "rsvd_scale": 0., "dataset": "boolq", "seed": 42,
              "train_batch_size": 8, "lora_alpha": 2, "lora_dropout": .05,
              "lora_target_modules": ["q_proj"]}
    count = [0]
    lock = threading.Lock()
    def guard(**kwargs):
        return None
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        def worker(index):
            groups = [{"params": [torch.nn.Parameter(torch.zeros(2, 3)),
                                  torch.nn.Parameter(torch.zeros(4, 2))]}]
            def generate():
                with lock:
                    count[0] += 1
                time.sleep(.05)
                with torch.no_grad():
                    groups[0]["params"][0].fill_(index + 1)
                    groups[0]["params"][1].fill_(index + 2)
            record = initial_factors(root, config, groups, generate, guard)
            return record, [p.detach().clone() for p in groups[0]["params"]]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(worker, range(4)))
        assert count[0] == 1
        assert sum(record["status"] == "generated" for record, _ in results) == 1
        assert len({record["adapter_sha256"] for record, _ in results}) == 1
        assert all(torch.equal(pair[0], results[0][1][0]) and torch.equal(pair[1], results[0][1][1])
                   for _, pair in results)
        path = root / "initializations/init-boolq-s42.pt"
        payload = torch.load(path, weights_only=True)
        payload["factors"][0][1] = torch.zeros(5, 2)
        torch.save(payload, path)
        groups = [{"params": [torch.nn.Parameter(torch.full((2, 3), -1.)),
                              torch.nn.Parameter(torch.full((4, 2), -1.))]}]
        try:
            initial_factors(root, config, groups, lambda: None, guard)
        except RuntimeError:
            assert all(torch.all(parameter == -1) for parameter in groups[0]["params"])
        else:
            raise AssertionError("Malformed cache was accepted")
    return {"status": "passed", "concurrent_workers": 4, "generations": count[0],
            "exact_reuse": True, "validation_before_copy": True}


if __name__ == "__main__":
    print("INITIALIZATION_SELF_TEST " + json.dumps(self_test()), flush=True)
