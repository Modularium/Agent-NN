from __future__ import annotations

"""Utility helpers for digital signatures."""

import json
import os
from pathlib import Path
from typing import Dict

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
)
from cryptography.exceptions import InvalidSignature

KEY_DIR = Path(os.getenv("KEY_DIR", "keys"))


def _priv_path(agent_id: str) -> Path:
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    return KEY_DIR / f"{agent_id}.pem"


def _pub_path(agent_id: str) -> Path:
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    return KEY_DIR / f"{agent_id}.pub"


def generate_keypair(agent_id: str) -> None:
    """Create a new Ed25519 keypair for ``agent_id``."""
    private = Ed25519PrivateKey.generate()
    priv_bytes = private.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    )
    with open(_priv_path(agent_id), "wb") as fh:
        fh.write(priv_bytes)
    public = private.public_key()
    pub_bytes = public.public_bytes(
        encoding=Encoding.PEM,
        format=PublicFormat.SubjectPublicKeyInfo,
    )
    with open(_pub_path(agent_id), "wb") as fh:
        fh.write(pub_bytes)


def _load_private(agent_id: str) -> Ed25519PrivateKey:
    with open(_priv_path(agent_id), "rb") as fh:
        data = fh.read()
    return serialization.load_pem_private_key(data, password=None)


def _load_public(agent_id: str) -> Ed25519PublicKey:
    with open(_pub_path(agent_id), "rb") as fh:
        data = fh.read()
    return serialization.load_pem_public_key(data)


def sign_payload(agent_id: str, payload: Dict) -> Dict:
    """Return signature info for ``payload`` signed by ``agent_id``."""
    priv = _load_private(agent_id)
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    signature = priv.sign(data)
    return {"signed_by": agent_id, "signature": signature.hex()}


def verify_signature(agent_id: str, payload: Dict, signature: str) -> bool:
    """Verify ``signature`` against ``payload`` for ``agent_id``."""
    try:
        pub = _load_public(agent_id)
    except FileNotFoundError:
        return False
    try:
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        pub.verify(bytes.fromhex(signature), data)
        return True
    except (InvalidSignature, ValueError):
        return False
