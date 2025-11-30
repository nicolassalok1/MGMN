from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet


ROOT = Path(__file__).resolve().parent
KEY_FILE = ROOT / ".gmgn_secret_key"
SECRET_FILE = ROOT / ".gmgn_secret"
HASH_FILE = ROOT / ".gmgn_secret_hash"


def generate_keyfile_if_missing() -> None:
    if not KEY_FILE.exists():
        KEY_FILE.write_bytes(Fernet.generate_key())


def _load_fernet() -> Optional[Fernet]:
    if not KEY_FILE.exists():
        return None
    try:
        key = KEY_FILE.read_bytes()
        return Fernet(key)
    except Exception:
        return None


def encrypt_and_store_api_key(api_key: str) -> None:
    if not api_key:
        return
    generate_keyfile_if_missing()
    f = _load_fernet()
    if f is None:
        return
    token = f.encrypt(api_key.encode("utf-8"))
    SECRET_FILE.write_bytes(token)
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    HASH_FILE.write_text(digest, encoding="utf-8")


def load_api_key() -> Optional[str]:
    f = _load_fernet()
    if f is None or not SECRET_FILE.exists():
        return None
    try:
        token = SECRET_FILE.read_bytes()
        return f.decrypt(token).decode("utf-8")
    except Exception:
        return None


def load_api_key_hash() -> Optional[str]:
    if not HASH_FILE.exists():
        return None
    try:
        return HASH_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def clear_api_key() -> None:
    for path in (SECRET_FILE, HASH_FILE):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
