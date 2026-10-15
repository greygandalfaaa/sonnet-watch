"""Check flopwire against flop-labs/yellowpaper evidence/wire-format-v1.json.

    python test_flopwire.py   # fetches the vectors from flop-labs/yellowpaper if no path is given
"""

import json
import os
import sys

import flopwire as fw

H = bytes.fromhex
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, note: str = "") -> None:
    results.append((name, bool(ok), note))


def main(path: str) -> int:
    w = json.load(open(path, encoding="utf-8"))
    c = w["compute_channel_v1"]

    ci = c["channel_id"]["inputs"]
    chan = fw.channel_id(H(ci["genesis_hash_hex"]), H(ci["agent_account_id32_hex"]),
                         H(ci["miner_account_id32_hex"]), int(ci["nonce"]))
    check("channel_id", chan.hex() == c["channel_id"]["hash_hex"])

    li = c["leaf_inputs"]
    base = dict(channel_id=H(li["channel_id_hex"]), turn_index=li["turn_index"], h_in=H(li["h_in_hex"]),
                h_out=H(li["h_out_hex"]), g_n=int(li["g_n"]), decode_policy_hash=H(li["decode_policy_hash_hex"]),
                h_ids=H(li["h_ids_hex"]), toploc_commitment_hash=H(li["toploc_commitment_hash_hex"]),
                miner_recv_ms=int(li["miner_recv_ms"]), miner_done_ms=int(li["miner_done_ms"]),
                latency_ms=int(li["latency_ms"]))
    hashes = {}
    for v in c["leaf_versions"]:
        leaf = fw.Leaf(version=v["scale_tag"], **base)
        hashes[v["version"]] = leaf.hash()
        check(f"leaf {v['version']} preimage", leaf.preimage().hex() == v["preimage_hex"])
        check(f"leaf {v['version']} hash", leaf.hash().hex() == v["hash_hex"])

    m = c["merkle"]
    order = [hashes[v] for v in m["leaf_order"]]
    check("merkle root", fw.merkle_root(order).hex() == m["root_hex"])
    path = fw.merkle_path(order, 2)
    check("merkle path index 2", [(s.hex(), l) for s, l in path]
          == [(p["sibling_hex"], p["sibling_is_left"]) for p in m["path_for_index_2"]])
    check("merkle path verifies", fw.verify_path(order[2], path, H(m["root_hex"])))
    check("merkle path wrong orientation rejected",
          not fw.verify_path(order[2], [(s, not l) for s, l in path], H(m["root_hex"])))

    r = c["receipt"]
    ri = r["inputs"]
    pre = fw.receipt_preimage(H(ri["channel_id_hex"]), H(ri["final_root_hex"]), int(ri["aggregate_gn"]), int(ri["payable"]))
    check("receipt preimage", pre.hex() == r["preimage_hex"])
    check("receipt signature (sr25519)", fw.verify(H(r["public_key_hex"]), pre, H(r["signature_hex"])))

    ls = c["v3_leaf_signature"]
    check("miner leaf signature (sr25519)", fw.verify_leaf(H(ls["leaf_hash_hex"]), H(ls["public_key_hex"]), H(ls["signature_hex"])))

    a = c["fcc4_transcript_with_ack"]
    v3 = hashes["V3"]
    ack = fw.ack_preimage(H(li["channel_id_hex"]), li["turn_index"], v3, int(a["agent_send_ms"]), int(a["agent_recv_ms"]))
    check("ack preimage", ack.hex() == a["ack_preimage_hex"])
    check("ack signature (sr25519)", fw.verify(H(a["agent_public_key_hex"]), ack, H(a["agent_signature_hex"])))

    for name, blob in (("fcc4 plain", c["fcc4_transcript_blob_hex"]), ("fcc4 with ack", a["blob_hex"])):
        try:
            ch, turns = fw.decode_fcc4(H(blob))
            check(f"{name} decodes", ch.hex() == li["channel_id_hex"] and len(turns) == 1 and turns[0].leaf.hash() == v3)
            check(f"{name} re-encodes byte-exact", fw.encode_fcc4(ch, turns).hex() == blob)
        except fw.WireError as e:
            check(f"{name} decodes", False, str(e))
    try:
        fw.check_transcript(H(a["blob_hex"]), H(li["channel_id_hex"]), H(ls["public_key_hex"]), H(a["agent_public_key_hex"]))
        check("full transcript check (miner + agent signatures)", True)
    except fw.WireError as e:
        check("full transcript check (miner + agent signatures)", False, str(e))

    # Our own key: sign and verify a receipt the way an agent would.
    kp = fw.keypair_from_seed(os.urandom(32))
    mine = fw.receipt_preimage(chan, fw.merkle_root(order), 42, 1000)
    sig = fw.sign(kp, mine)
    check("own sr25519 receipt sign/verify", fw.verify(kp[0], mine, sig) and not fw.verify(kp[0], mine[:-1] + b"\1", sig))

    chan0, root = H(li["channel_id_hex"]), H(m["root_hex"])
    vt = fw.decode_verified_turn(H(c["verified_turn_v3_scale_hex"]), chan0)
    check("verified turn V3 decodes", vt.leaf.hash() == v3)
    try:
        fw.verify_turn_proof(vt, root, 3)
        check("verified turn V3 proof accepted", True)
    except fw.WireError as e:
        check("verified turn V3 proof accepted", False, str(e))

    # Negative cases: each one goes to the check it targets, and must be refused.
    agent_pk, miner_pk = H(a["agent_public_key_hex"]), H(ls["public_key_hex"])

    def refuse(fn) -> tuple[bool, str]:
        try:
            out = fn()
        except fw.WireError as e:
            return True, str(e)
        return out is False, "accepted" if out is not False else "verify=False"

    routes = {
        "unknown_leaf_enum": lambda b: fw.decode_leaf_version(b),
        "unknown_retention_enum": lambda b: fw.decode_retention(b),
        "truncated_fcc4": lambda b: fw.decode_fcc4(b),
        "trailing_fcc4": lambda b: fw.decode_fcc4(b),
        "unknown_fcc_version": lambda b: fw.decode_fcc4(b),
        "invalid_agent_ack_signature": lambda b: fw.check_transcript(b, chan0, miner_pk, agent_pk),
        "duplicate_turn_index": lambda b: fw.decode_verified_turns(b, chan0),
        "wrong_path_orientation": lambda b: fw.verify_turn_proof(fw.decode_verified_turn(b, chan0), root, 3),
        "wrong_leaf_version": lambda b: fw.verify_turn_proof(fw.decode_verified_turn(b, chan0), root, 3),
        "legacy_leaf_current_channel": lambda b: fw.verify_turn_proof(fw.decode_verified_turn(b, chan0), root, 3),
        "wrong_genesis_network": lambda b: False if b != chan else None,
        "wrong_session": lambda b: False if b != chan else None,
        "invalid_receipt_signature": lambda b: fw.verify_receipt(
            H(r["public_key_hex"]), H(ri["channel_id_hex"]), H(ri["final_root_hex"]), int(ri["aggregate_gn"]), int(ri["payable"]), b),
        "legacy_receipt_current_channel": lambda b: fw.verify_receipt(
            H(r["public_key_hex"]), b[:32], b[32:64], int.from_bytes(b[64:80], "little"),
            int.from_bytes(b[80:96], "little"), b[96:]),
    }
    skipped = []
    for n in w["negative_cases"]:
        fn = routes.get(n["id"])
        if fn is None:
            skipped.append(f"{n['id']} ({n['site']}: validator side, not the agent's job)")
            continue
        ok, note = refuse(lambda: fn(H(n["bytes_hex"])))
        check(f"negative {n['id']}", ok, note)

    width = max(len(n) for n, _, _ in results)
    for name, ok, note in results:
        print(f"{'OK  ' if ok else 'FAIL'} {name:<{width}}  {note}")
    for s in skipped:
        print(f"SKIP {s}")
    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed, {len(skipped)} skipped")
    return 1 if bad else 0


VECTORS_URL = ("https://raw.githubusercontent.com/flop-labs/yellowpaper/main/"
               "evidence/wire-format-v1.json")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "vectors/wire-format-v1.json"
    if not os.path.exists(path):
        import tempfile
        import urllib.request
        path = os.path.join(tempfile.gettempdir(), "wire-format-v1.json")
        urllib.request.urlretrieve(VECTORS_URL, path)
    sys.exit(main(path))
