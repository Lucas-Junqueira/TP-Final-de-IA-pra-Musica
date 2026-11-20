"""Testes de src/tokenize.py com MIDIs sintéticos."""

import pytest
from symusic import Note, Score, Tempo, TimeSignature, Track

from src.tokenize import (
    build_tokenizer,
    decode_sequence,
    emopia_splits,
    encode_events,
    load_score,
    make_windows,
    parse_emopia_name,
    pop1k7_splits,
    special_ids,
    to_sequence,
)

TOK = build_tokenizer()
SP = special_ids(TOK)
BAR = TOK.vocab["Bar_None"]


def write_midi(path, n_bars=8, num=4, n_tracks=2):
    s = Score(480)
    s.time_signatures.append(TimeSignature(0, num, 4))
    s.tempos.append(Tempo(0, 100.0))
    for k in range(n_tracks):
        t = Track(name=f"t{k}", program=0)
        for i in range(n_bars * 4):  # uma semínima por tempo
            t.notes.append(Note(i * 480, 480, 48 + 12 * k + i % 12, 80))
        s.tracks.append(t)
    s.dump_midi(str(path))
    return path


def test_vocab_and_special_ids():
    assert SP.pad == 0
    assert len(set([SP.bos, SP.eos, SP.pad, SP.null, *SP.emotions])) == 8
    kinds = {t.split("_")[0] for t in TOK.vocab}
    assert {"Bar", "Position", "Pitch", "Velocity", "Duration", "Tempo"} <= kinds
    assert sum(t.startswith("Position") for t in TOK.vocab) == 16
    assert sum(t.startswith("Pitch_") for t in TOK.vocab) == 88


def test_roundtrip_keeps_notes(tmp_path):
    score = load_score(write_midi(tmp_path / "a.mid"))
    assert len(score.tracks) == 1  # trilhas juntadas
    seq = to_sequence(encode_events(TOK, score), SP.emotions[2], SP)
    assert seq[:2] == [SP.bos, SP.emotions[2]] and seq[-1] == SP.eos
    back, dropped = decode_sequence(TOK, seq, SP)
    assert dropped == 0
    orig = sorted((n.pitch, n.velocity) for n in score.tracks[0].notes)
    new = sorted((n.pitch, n.velocity) for n in back.tracks[0].notes)
    assert len(orig) == len(new) == 64
    assert [p for p, _ in orig] == [p for p, _ in new]


def test_rejects_non_4_4(tmp_path):
    assert load_score(write_midi(tmp_path / "b.mid", num=3)) is None


def test_decode_drops_invalid_tokens(tmp_path):
    score = load_score(write_midi(tmp_path / "c.mid", n_bars=2))
    seq = to_sequence(encode_events(TOK, score), SP.null, SP)
    seq.insert(3, TOK.vocab["Velocity_63"])  # fora de ordem
    _, dropped = decode_sequence(TOK, seq, SP)
    assert dropped == 1


def test_windows_start_at_bar_with_prefix(tmp_path):
    score = load_score(write_midi(tmp_path / "d.mid", n_bars=40))
    seq = to_sequence(encode_events(TOK, score), SP.emotions[0], SP)
    windows = make_windows(seq, max_len=128, bar_id=BAR)
    assert len(windows) > 1
    for w in windows:
        assert len(w) <= 128
        assert w[:3] == [SP.bos, SP.emotions[0], BAR]
    # sem perder tokens: o corpo das janelas cobre a sequência inteira
    assert sum(len(w) - 2 for w in windows) == len(seq) - 2
    assert make_windows(seq, max_len=10_000, bar_id=BAR) == [seq]


def test_parse_emopia_name():
    assert parse_emopia_name("Q1__8v0MFBZoco_0.mid") == ("Q1", "_8v0MFBZoco")
    assert parse_emopia_name("Q3_0EWWyyV7RJA_12.mid") == ("Q3", "0EWWyyV7RJA")


def test_emopia_split_overlap_is_detected(tmp_path):
    (tmp_path / "split").mkdir()
    for split, names in [("train", ["Q1_aaa_0.mid"]), ("val", ["Q2_bbb_0.mid"]),
                         ("test", ["Q1_aaa_1.mid"])]:
        (tmp_path / "split" / f"{split}_clip.csv").write_text(
            ",clip_name\n" + "".join(f"{i},{n}\n" for i, n in enumerate(names)))
    with pytest.raises(ValueError, match="aaa"):
        emopia_splits(tmp_path)


def test_pop1k7_split(tmp_path):
    for i in range(40):
        (tmp_path / "src_001").mkdir(exist_ok=True)
        (tmp_path / "src_001" / f"{i}.mid").touch()
    s = pop1k7_splits(tmp_path)
    assert len(s["val"]) == 2 and len(s["train"]) == 38
    assert not set(s["val"]) & set(s["train"])
    assert s == pop1k7_splits(tmp_path)  # determinística
    s2 = pop1k7_splits(tmp_path, exclude={"src_001/3.mid"})
    assert "src_001/3.mid" not in s2["train"] + s2["val"]
    assert len(s2["train"]) + len(s2["val"]) == 39
