"""Tokenizer tests: roundtrips, determinism, specials, edge cases.

Uses a tiny vocab (fast) trained on a small in-memory corpus, plus
save/load roundtrip. Full 8192-vocab training is timed in train_tokenizer.py.
"""
import os
import random

from tokenizer.bpe import BPETokenizer, EOT, EOT_ID, PAD, PAD_ID

SMALL_CORPUS = [
    "Once upon a time there was a little dragon. ",
    "The dragon was sad and wanted a friend. ",
    "Lily laughed and played in the green hills! 123 ",
    "   whitespace\n\ttest  ",
    "Dragons love cake. " * 5,
]


def make_small_tok():
    tok = BPETokenizer()
    tok.train(list(SMALL_CORPUS), vocab_size=300, verbose=False)
    return tok


def test_roundtrip_ascii():
    tok = make_small_tok()
    for s in ["hello world", "Once upon a time...", "123 dragons!"]:
        assert tok.decode(tok.encode(s)) == s


def test_roundtrip_unicode_emoji():
    tok = make_small_tok()
    for s in ["naïve café 🐉🔥", "日本語テスト", "e\u0301 mixed café"]:
        assert tok.decode(tok.encode(s)) == s


def test_whitespace_edge_cases():
    tok = make_small_tok()
    for s in ["", " ", "   ", "\n", "\t", "a\n\nb", "  leading and trailing  ",
              "line1\nline2\tend"]:
        assert tok.decode(tok.encode(s)) == s


def test_empty_string():
    tok = make_small_tok()
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_determinism_train_twice_same_merges():
    t1 = BPETokenizer()
    t1.train(list(SMALL_CORPUS), vocab_size=300, verbose=False)
    t2 = BPETokenizer()
    t2.train(list(SMALL_CORPUS), vocab_size=300, verbose=False)
    assert t1.merges == t2.merges
    s = "The sad dragon loved cake."
    assert t1.encode(s) == t2.encode(s)


def test_special_token_handling():
    tok = make_small_tok()
    ids = tok.encode(f"hello{EOT}world")
    assert EOT_ID in ids
    assert tok.decode(ids) == f"hello{EOT}world"
    ids2 = tok.encode(f"a{PAD}b")
    assert PAD_ID in ids2
    assert tok.decode(ids2) == f"a{PAD}b"


def test_special_string_inside_training_text():
    # A literal "<|endoftext|>" in training data must not corrupt merges,
    # and must still encode as the special id afterwards.
    corpus = list(SMALL_CORPUS) + [f"story break {EOT} new story"]
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=300, verbose=False)
    ids = tok.encode(f"x{EOT}y")
    assert EOT_ID in ids
    assert tok.decode(ids) == f"x{EOT}y"


def test_random_bytes_roundtrip(tmp_path):
    # Random unicode strings (multi-byte heavy) must roundtrip exactly.
    rng = random.Random(0)
    tok = make_small_tok()
    for _ in range(20):
        s = "".join(chr(rng.randint(1, 0x2FFF)) for _ in range(rng.randint(0, 40)))
        assert tok.decode(tok.encode(s)) == s
    # Raw byte ids 0-255 decode, then re-encode+decode is stable.
    ids = [rng.randint(0, 255) for _ in range(64)]
    once = tok.decode(ids)
    assert tok.decode(tok.encode(once)) == once


def test_save_load_roundtrip(tmp_path):
    tok = make_small_tok()
    p = os.path.join(tmp_path, "tok.json")
    tok.save(p)
    t2 = BPETokenizer.load(p)
    assert t2.merges == tok.merges
    for s in ["dragons 🐉", f"a{EOT}b", "  spaces\n"]:
        assert t2.decode(t2.encode(s)) == s
