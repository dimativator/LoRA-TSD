"""Unpack an authenticated private runtime on the allocated cloud worker."""

import io
import os
from pathlib import Path
import zipfile

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


payload = Path(__file__).with_name("siqa_runtime.enc").read_bytes()
key = bytes.fromhex(os.environ.pop("LORA_SIQA_BUNDLE_KEY"))
plaintext = AESGCM(key).decrypt(payload[:12], payload[12:], b"lora-siqa-retractions-v1")
target = Path("/tmp/private-siqa-runtime")
target.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(plaintext)) as archive:
    for name in archive.namelist():
        if not (target / name).resolve().is_relative_to(target.resolve()):
            raise ValueError("Invalid archive path")
    archive.extractall(target)
print(f"Authenticated runtime unpacked: {len(plaintext)} compressed bytes")
