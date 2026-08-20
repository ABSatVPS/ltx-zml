"""Minimal byte-level BPE tokenizer.json for the synthetic fixtures (vocab 8192).

Byte-level only: the 256 GPT-2-style byte tokens plus filler ids so the vocab
size matches the fixture's config (random weights, so this is purely so the
engine and server can round-trip text — not a real tokenizer)."""
import json, sys
from pathlib import Path

out = Path(sys.argv[1]); vocab_size = int(sys.argv[2])

# GPT-2 byte->unicode map (same table HF byte-level BPE uses)
bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("\xa1"), ord("\xac")+1)) + list(range(ord("\xae"), ord("\xff")+1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256+n); n += 1
byte_to_uni = {b: chr(c) for b, c in zip(bs, cs)}

vocab = {}
for b in range(256):
    vocab[byte_to_uni[b]] = len(vocab)
# filler tokens to reach vocab_size (never produced by encode; ids must exist
# because the model's lm_head has vocab_size rows)
i = 0
while len(vocab) < vocab_size - 3:
    tok = f"Āfill{i}"          # prefix keeps them out of byte space
    if tok not in vocab: vocab[tok] = len(vocab)
    i += 1
specials = ["<|endoftext|>", "<|user|>", "<|observation|>"]
added = []
for s in specials:
    vocab[s] = len(vocab)
    added.append({"id": vocab[s], "content": s, "single_word": False, "lstrip": False,
                  "rstrip": False, "normalized": False, "special": True})

tok = {
    "version": "1.0", "truncation": None, "padding": None,
    "added_tokens": added,
    "normalizer": None,
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
    "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
    "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
    "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
              "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
              "ignore_merges": True, "vocab": vocab, "merges": []},
}
out.write_text(json.dumps(tok))
print(f"wrote {out} — {len(vocab)} tokens ({len(added)} special)")
