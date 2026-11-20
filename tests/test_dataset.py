"""Testes de src/dataset.py."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import PitchShifter, WindowDataset, collate, load_split, make_loader
from src.tokenize import build_tokenizer, special_ids

TOK = build_tokenizer()
SP = special_ids(TOK)
V = TOK.vocab
BAR = V["Bar_None"]


def bar(pitches, pos=0):
    out = [BAR, V[f"Position_{pos}"]]
    for p in pitches:
        out += [V[f"Pitch_{p}"], V["Velocity_63"], V["Duration_1.0.4"]]
    return out


def piece(n_bars, cond, pitches=(60, 64, 67)):
    body = [t for _ in range(n_bars) for t in bar(pitches)]
    return {"ids": np.array([SP.bos, cond, *body, SP.eos], dtype=np.int16)}


def pitch_values(ids):
    return [int(TOK[int(i)].split("_")[1]) for i in ids if TOK[int(i)].startswith("Pitch_")]


def test_shift_changes_only_pitches():
    sh = PitchShifter(V)
    ids = bar([60, 64])
    out = sh.shift(ids, 2)
    assert pitch_values(out) == [62, 66]
    other = [i for i, t in enumerate(ids) if not TOK[t].startswith("Pitch_")]
    assert all(out[i] == ids[i] for i in other)


def test_allowed_shifts_respect_piano_range():
    sh = PitchShifter(V)
    assert sh.allowed_shifts(bar([22, 60])) == [-1, 0, 1, 2, 3]
    assert sh.allowed_shifts(bar([60, 107])) == [-3, -2, -1, 0, 1]
    assert sh.allowed_shifts([BAR]) == [0]
    out = sh.random_transpose(bar([21]), random.Random(0))
    assert min(pitch_values(out)) >= 21


def test_train_windows():
    torch.manual_seed(0)
    pieces = [piece(200, SP.emotions[1]), piece(5, SP.emotions[3])]
    ds = WindowDataset(pieces, SP, V, max_len=128, train=True)
    assert len(ds) == sum(max(1, -(-(len(p["ids"]) - 2) // 126)) for p in pieces)
    for i in range(len(ds)):
        x = ds[i]
        assert x.dtype == torch.long and len(x) <= 128
        assert x[0] == SP.bos and x[1] in (ds.pieces[ds.index[i]][1], SP.null)
        assert x[2] == BAR
        assert all(21 <= p <= 108 for p in pitch_values(x))


def test_condition_dropout_rate():
    torch.manual_seed(0)
    ds = WindowDataset([piece(3, SP.emotions[0])], SP, V, max_len=64, train=True)
    conds = [int(ds[0][1]) for _ in range(4000)]
    rate = conds.count(SP.null) / len(conds)
    assert 0.12 < rate < 0.18
    assert set(conds) == {SP.emotions[0], SP.null}


def test_transposition_happens_in_train_only():
    torch.manual_seed(0)
    p = [piece(3, SP.emotions[0])]
    train = WindowDataset(p, SP, V, max_len=64, train=True)
    shifts = {pitch_values(train[0])[0] - 60 for _ in range(300)}
    assert shifts == {-3, -2, -1, 0, 1, 2, 3}
    val = WindowDataset(p, SP, V, max_len=64, train=False)
    assert torch.equal(val[0], torch.from_numpy(p[0]["ids"].astype(np.int64)))


def test_val_windows_are_deterministic_and_cover_piece():
    p = piece(100, SP.emotions[2])
    ds = WindowDataset([p], SP, V, max_len=128, train=False)
    assert len(ds) > 1
    assert all(ds[i][:3].tolist() == [SP.bos, SP.emotions[2], BAR] for i in range(len(ds)))
    assert sum(len(ds[i]) - 2 for i in range(len(ds))) == len(p["ids"]) - 2


def test_collate_and_loader():
    batch = collate([torch.tensor([1, 4, 8]), torch.tensor([1, 5, 8, 9, 2])], pad_id=SP.pad)
    assert batch["tokens"].shape == (2, 5)
    assert batch["mask"].tolist() == [[True] * 3 + [False] * 2, [True] * 5]
    loader = make_loader([piece(50, SP.emotions[0]), piece(7, SP.emotions[1])], SP, V,
                         batch_size=4, train=True, max_len=64)
    b = next(iter(loader))
    assert b["tokens"].shape[1] <= 64 and (b["tokens"][:, 0] == SP.bos).all()


@pytest.mark.skipif(not Path("Data/tokens/emopia_train.pkl").exists(), reason="dados não tokenizados")
def test_real_data_smoke():
    torch.manual_seed(0)
    loader = make_loader(load_split("emopia", "train"), SP, V, batch_size=8, train=True)
    b = next(iter(loader))
    assert b["tokens"].shape[0] == 8 and b["tokens"].shape[1] <= 1024
    assert set(b["tokens"][:, 1].tolist()) <= {*SP.emotions, SP.null}
