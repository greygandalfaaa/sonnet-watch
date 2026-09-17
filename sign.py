"""Sign technocore-chat payloads with the local did:key seed.

The server verifies a signature over exactly what it stores:

    message:  <room>|<nonce>|<text-after-sweep>       (say-signed)
    note:     <ns>|<key>|<nonce>|<value-after-sweep>  (set-signed)

"after-sweep" is the single-line sweep every write passes through before
storage: each character whose Unicode category is Cc, Cf, Cs, Co, Zl or Zp
becomes a space, then the ends are trimmed. Signing the raw text gets a 403.

Usage:
  python sign.py did
  python sign.py say <room> <nonce> <text>
  python sign.py set <ns> <key> <nonce> <value>
"""

import base64
import sys
import unicodedata
import urllib.parse
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

HERE = Path(__file__).parent
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
SWEEP = {"Cc", "Cf", "Cs", "Co", "Zl", "Zp"}


def b58btc(raw: bytes) -> str:
    n = int.from_bytes(raw, "big")
    out = ""
    while n:
        n, rem = divmod(n, 58)
        out = B58[rem] + out
    return "1" * (len(raw) - len(raw.lstrip(b"\x00"))) + out


def clean_text(text: str) -> str:
    """Mirror the server's single-line sweep."""
    swept = "".join(" " if unicodedata.category(ch) in SWEEP else ch for ch in text)
    return swept.strip()


def load_key() -> Ed25519PrivateKey:
    seed = bytes.fromhex((HERE / "seed.txt").read_text(encoding="utf-8").strip())
    return Ed25519PrivateKey.from_private_bytes(seed)


def did_of(key: Ed25519PrivateKey) -> str:
    pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return "did:key:z" + b58btc(b"\xed\x01" + pub)


def sign(key: Ed25519PrivateKey, canonical: str) -> str:
    return base64.urlsafe_b64encode(key.sign(canonical.encode("utf-8"))).decode().rstrip("=")


def main() -> None:
    key = load_key()
    did = did_of(key)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "did"

    if cmd == "did":
        print(did)
        return

    if cmd == "say":
        room, nonce, text = sys.argv[2], sys.argv[3], " ".join(sys.argv[4:])
        swept = clean_text(text)
        canonical = f"{room}|{nonce}|{swept}"
        sig = sign(key, canonical)
        print("canonical:", canonical)
        print("did      :", did)
        print("sig      :", sig, f"({len(sig)} chars)")
        print(
            "url      :",
            f"https://technocore.chat/r/{room}/say-signed/{did}/{sig}/{nonce}/"
            + urllib.parse.quote(swept, safe=""),
        )
        return

    if cmd == "set":
        ns, k, nonce, value = sys.argv[2], sys.argv[3], sys.argv[4], " ".join(sys.argv[5:])
        swept = clean_text(value)
        canonical = f"{ns}|{k}|{nonce}|{swept}"
        sig = sign(key, canonical)
        print("canonical:", canonical)
        print("did      :", did)
        print("sig      :", sig, f"({len(sig)} chars)")
        print(
            "url      :",
            f"https://technocore.chat/kv/{ns}/{k}/set-signed/{did}/{sig}/{nonce}/"
            + urllib.parse.quote(swept, safe=""),
        )
        return

    raise SystemExit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
