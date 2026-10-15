"""Agent side of the Flop compute channel (flop-wire-v1, Yellow Paper Appendix F).

An agent buying inference on Flop does four things this module covers:
derive the channel id it opens, check each turn the miner returns (leaf hash and
the miner's signature), acknowledge each turn with its own signature, and at the
end sign the receipt whose `payable` authorises the miner's payout. The receipt
is what settles a session, and settled payables are what the Yellow Paper (R8.4)
scores agents on, so getting these bytes right is the whole job.

Signatures are sr25519 over the raw preimage; hashes are BLAKE2b-256. Integers
are little-endian (SCALE fixed width); version tags and options are one byte.
Everything here is checked against the public vectors in
flop-labs/yellowpaper evidence/wire-format-v1.json (see test_flopwire.py).

Chain calls (open_channel, settle) and the miner transport are not here: the
testnet RPC and miner endpoints do not exist yet.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import sr25519

CHANNEL_DOMAIN = b"FLOP/COMPUTE_CHANNEL/ID"
RECEIPT_DOMAIN = b"FLOP/COMPUTE_CHANNEL/RECEIPT"
PROTOCOL_VERSION = 1
FCC4_MAGIC = b"FCC4"
LEAF_VERSIONS = (0, 1, 2, 3)
RETENTION_TAGS = (0, 1)


class WireError(ValueError):
    """Bytes that a conforming agent must refuse."""


def blake2_256(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=32).digest()


def u32(n: int) -> bytes:
    return struct.pack("<I", n)


def u64(n: int) -> bytes:
    return struct.pack("<Q", n)


def u128(n: int) -> bytes:
    return n.to_bytes(16, "little")


def h32(b: bytes, what: str) -> bytes:
    if len(b) != 32:
        raise WireError(f"{what}: expected 32 bytes, got {len(b)}")
    return b


# ---------- channel id ----------

def channel_id(genesis_hash: bytes, agent: bytes, miner: bytes, nonce: int) -> bytes:
    """blake2_256(domain || version || genesis || agent || miner || nonce:u64)."""
    pre = (CHANNEL_DOMAIN + bytes([PROTOCOL_VERSION]) + h32(genesis_hash, "genesis")
           + h32(agent, "agent") + h32(miner, "miner") + u64(nonce))
    return blake2_256(pre)


# ---------- leaves ----------

@dataclass(frozen=True)
class Leaf:
    version: int
    channel_id: bytes
    turn_index: int
    h_in: bytes
    h_out: bytes
    g_n: int
    decode_policy_hash: bytes = b"\0" * 32
    h_ids: bytes = b"\0" * 32
    toploc_commitment_hash: bytes = b"\0" * 32
    miner_recv_ms: int = 0
    miner_done_ms: int = 0
    latency_ms: int = 0

    def preimage(self) -> bytes:
        if self.version not in LEAF_VERSIONS:
            raise WireError(f"unknown leaf version {self.version}")
        p = h32(self.channel_id, "channel_id") + u32(self.turn_index) + h32(self.h_in, "h_in") \
            + h32(self.h_out, "h_out") + u128(self.g_n)
        if self.version >= 2:
            p += self.decode_policy_hash
        if self.version == 3:
            p += self.h_ids + self.toploc_commitment_hash
        if self.version >= 1:
            p += u64(self.miner_recv_ms) + u64(self.miner_done_ms) + u64(self.latency_ms)
        return p

    def hash(self) -> bytes:
        return blake2_256(self.preimage())


# ---------- merkle ----------

def merkle_root(leaf_hashes: list[bytes]) -> bytes:
    """Node = blake2_256(left || right), no prefix; an odd node is paired with itself."""
    if not leaf_hashes:
        raise WireError("no leaves")
    level = list(leaf_hashes)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [blake2_256(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_path(leaf_hashes: list[bytes], index: int) -> list[tuple[bytes, bool]]:
    """Siblings from leaf to root, each with whether the sibling sits on the left."""
    level, path, i = list(leaf_hashes), [], index
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sib = i ^ 1
        path.append((level[sib], sib < i))
        level = [blake2_256(level[j] + level[j + 1]) for j in range(0, len(level), 2)]
        i //= 2
    return path


def verify_path(leaf_hash: bytes, path: list[tuple[bytes, bool]], root: bytes) -> bool:
    node = leaf_hash
    for sib, sib_is_left in path:
        node = blake2_256(sib + node) if sib_is_left else blake2_256(node + sib)
    return node == root


# ---------- signatures ----------

def keypair_from_seed(seed32: bytes):
    """sr25519 keypair (public, secret) from a 32-byte mini-secret."""
    return sr25519.pair_from_seed(seed32)


def sign(keypair, message: bytes) -> bytes:
    return sr25519.sign(keypair, message)


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    try:
        return sr25519.verify(signature, message, public)
    except Exception:  # noqa: BLE001 - malformed key or signature is simply invalid
        return False


def verify_leaf(leaf_hash: bytes, miner_public: bytes, miner_signature: bytes) -> bool:
    """The miner signs the leaf hash."""
    return verify(miner_public, leaf_hash, miner_signature)


def ack_preimage(chan: bytes, turn_index: int, leaf_hash: bytes, agent_send_ms: int, agent_recv_ms: int) -> bytes:
    return h32(chan, "channel_id") + u32(turn_index) + h32(leaf_hash, "leaf") + u64(agent_send_ms) + u64(agent_recv_ms)


def receipt_preimage(chan: bytes, final_root: bytes, aggregate_gn: int, payable: int) -> bytes:
    """What the agent signs to authorise a payout: domain || version || channel || root || G_n || payable."""
    return (RECEIPT_DOMAIN + bytes([PROTOCOL_VERSION]) + h32(chan, "channel_id")
            + h32(final_root, "root") + u128(aggregate_gn) + u128(payable))


# ---------- FCC4 transcript ----------

@dataclass(frozen=True)
class Turn:
    leaf: Leaf
    retention: int
    miner_signature: bytes
    ack: tuple[int, int, bytes] | None  # (agent_send_ms, agent_recv_ms, agent_signature)


class _Reader:
    def __init__(self, b: bytes):
        self.b, self.i = b, 0

    def take(self, n: int) -> bytes:
        if self.i + n > len(self.b):
            raise WireError("truncated")
        out = self.b[self.i:self.i + n]
        self.i += n
        return out

    def int_(self, n: int) -> int:
        return int.from_bytes(self.take(n), "little")


def decode_fcc4(blob: bytes) -> tuple[bytes, list[Turn]]:
    """FCC4 || channel_id || count:u32 || turns. Each turn: leaf version tag, leaf fields
    (retention tag after G_n), miner signature, then Option<agent ack>."""
    r = _Reader(blob)
    if r.take(4) != FCC4_MAGIC:
        raise WireError("unknown fcc version")
    chan = r.take(32)
    turns: list[Turn] = []
    seen: set[int] = set()
    for _ in range(r.int_(4)):
        version = r.int_(1)
        if version not in LEAF_VERSIONS:
            raise WireError("unknown leaf enum")
        idx = r.int_(4)
        if idx in seen:
            raise WireError("duplicate turn index")
        seen.add(idx)
        h_in, h_out, g_n = r.take(32), r.take(32), r.int_(16)
        retention = r.int_(1)
        if retention not in RETENTION_TAGS:
            raise WireError("unknown retention enum")
        dp = r.take(32) if version >= 2 else b"\0" * 32
        ids, toploc = (r.take(32), r.take(32)) if version == 3 else (b"\0" * 32, b"\0" * 32)
        recv, done, lat = (r.int_(8), r.int_(8), r.int_(8)) if version >= 1 else (0, 0, 0)
        leaf = Leaf(version, chan, idx, h_in, h_out, g_n, dp, ids, toploc, recv, done, lat)
        sig = r.take(64)
        tag = r.int_(1)
        if tag == 0:
            ack = None
        elif tag == 1:
            ack = (r.int_(8), r.int_(8), r.take(64))
        else:
            raise WireError("bad option tag")
        turns.append(Turn(leaf, retention, sig, ack))
    if r.i != len(blob):
        raise WireError("trailing bytes")
    return chan, turns


def encode_fcc4(chan: bytes, turns: list[Turn]) -> bytes:
    out = FCC4_MAGIC + h32(chan, "channel_id") + u32(len(turns))
    for t in turns:
        L = t.leaf
        out += bytes([L.version]) + u32(L.turn_index) + L.h_in + L.h_out + u128(L.g_n) + bytes([t.retention])
        if L.version >= 2:
            out += L.decode_policy_hash
        if L.version == 3:
            out += L.h_ids + L.toploc_commitment_hash
        if L.version >= 1:
            out += u64(L.miner_recv_ms) + u64(L.miner_done_ms) + u64(L.latency_ms)
        out += t.miner_signature
        out += b"\0" if t.ack is None else b"\1" + u64(t.ack[0]) + u64(t.ack[1]) + t.ack[2]
    return out


def decode_leaf_version(tag: bytes) -> int:
    if len(tag) != 1 or tag[0] not in LEAF_VERSIONS:
        raise WireError("unknown leaf enum")
    return tag[0]


def decode_retention(tag: bytes) -> int:
    if len(tag) != 1 or tag[0] not in RETENTION_TAGS:
        raise WireError("unknown retention enum")
    return tag[0]


def _compact(r: "_Reader") -> int:
    """SCALE compact integer (single, two and four byte modes)."""
    b0 = r.int_(1)
    mode = b0 & 3
    if mode == 0:
        return b0 >> 2
    if mode == 1:
        return (b0 | r.int_(1) << 8) >> 2
    if mode == 2:
        return (b0 | r.int_(3) << 8) >> 2
    raise WireError("compact big-integer mode not supported here")


@dataclass(frozen=True)
class VerifiedTurn:
    leaf: Leaf
    miner_signature: bytes
    path: list[tuple[bytes, bool]]


def _verified_turn(r: "_Reader", chan: bytes) -> VerifiedTurn:
    """SCALE VerifiedTurn: leaf tag, leaf fields (no retention), miner signature, Vec<(H256, bool)>."""
    version = decode_leaf_version(r.take(1))
    idx, h_in, h_out, g_n = r.int_(4), r.take(32), r.take(32), r.int_(16)
    dp = r.take(32) if version >= 2 else b"\0" * 32
    ids, toploc = (r.take(32), r.take(32)) if version == 3 else (b"\0" * 32, b"\0" * 32)
    recv, done, lat = (r.int_(8), r.int_(8), r.int_(8)) if version >= 1 else (0, 0, 0)
    sig = r.take(64)
    path = []
    for _ in range(_compact(r)):
        sib, left = r.take(32), r.int_(1)
        if left not in (0, 1):
            raise WireError("bad bool")
        path.append((sib, bool(left)))
    return VerifiedTurn(Leaf(version, chan, idx, h_in, h_out, g_n, dp, ids, toploc, recv, done, lat), sig, path)


def decode_verified_turn(b: bytes, chan: bytes) -> VerifiedTurn:
    r = _Reader(b)
    t = _verified_turn(r, chan)
    if r.i != len(b):
        raise WireError("trailing bytes")
    return t


def decode_verified_turns(b: bytes, chan: bytes) -> list[VerifiedTurn]:
    """Vec<VerifiedTurn>; a repeated turn index is refused (DuplicateVerifiedTurn)."""
    r = _Reader(b)
    turns, seen = [], set()
    for _ in range(_compact(r)):
        t = _verified_turn(r, chan)
        if t.leaf.turn_index in seen:
            raise WireError("duplicate turn index")
        seen.add(t.leaf.turn_index)
        turns.append(t)
    if r.i != len(b):
        raise WireError("trailing bytes")
    return turns


def verify_turn_proof(t: VerifiedTurn, root: bytes, channel_leaf_version: int) -> None:
    """A turn counts only under the channel's selected leaf version and inside the committed root."""
    if t.leaf.version != channel_leaf_version:
        raise WireError("unsupported leaf version for this channel")
    if not verify_path(t.leaf.hash(), t.path, root):
        raise WireError("leaf not in root")


def verify_receipt(public: bytes, chan: bytes, final_root: bytes, aggregate_gn: int, payable: int, signature: bytes) -> bool:
    return verify(public, receipt_preimage(chan, final_root, aggregate_gn, payable), signature)


def check_transcript(blob: bytes, expect_channel: bytes, miner_public: bytes, agent_public: bytes) -> list[Turn]:
    """Everything an agent must verify before signing a receipt over this transcript."""
    chan, turns = decode_fcc4(blob)
    if chan != expect_channel:
        raise WireError("wrong session")
    for t in turns:
        lh = t.leaf.hash()
        if not verify_leaf(lh, miner_public, t.miner_signature):
            raise WireError("invalid miner leaf signature")
        if t.ack is not None:
            send, recv, sig = t.ack
            if not verify(agent_public, ack_preimage(chan, t.leaf.turn_index, lh, send, recv), sig):
                raise WireError("invalid agent ack signature")
    return turns
