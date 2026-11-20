"""Testes de src/train.py com dados sintéticos."""

import csv
import pickle

import numpy as np
import pytest
import torch

from src.tokenize import build_tokenizer, special_ids
from src.train import _parse_value, load_config, load_model, lr_factor, train

TOK = build_tokenizer()
SP = special_ids(TOK)
V = TOK.vocab


def fake_piece(rng, cond, n_bars=6):
    body = []
    for _ in range(n_bars):
        body += [V["Bar_None"], V["Position_0"]]
        for p in rng.integers(48, 72, size=3):
            body += [V[f"Pitch_{p}"], V["Velocity_63"], V["Duration_1.0.4"]]
    return {"ids": np.array([SP.bos, cond, *body, SP.eos], dtype=np.int16)}


@pytest.fixture
def tokens_dir(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 12), ("val", 4)):
        pieces = [fake_piece(rng, SP.emotions[i % 4]) for i in range(n)]
        with open(tmp_path / f"emopia_{split}.pkl", "wb") as f:
            pickle.dump(pieces, f)
    return tmp_path


def tiny_cfg(tokens_dir, out_dir, **train_kw):
    return {
        "out_dir": str(out_dir),
        "model": {"type": "transformer", "n_layers": 1, "d_model": 32, "n_heads": 4,
                  "max_len": 64, "dropout": 0.0},
        "data": {"dataset": "emopia", "tokens_dir": str(tokens_dir), "batch_size": 4},
        "train": {"lr": 3e-3, "warmup_steps": 5, "max_steps": 30, "eval_every": 10,
                  "log_every": 10, "amp": False, "device": "cpu", **train_kw},
    }


def test_lr_schedule():
    assert lr_factor(0, warmup=10, total=100, min_ratio=0.1) == pytest.approx(0.1)
    assert lr_factor(9, warmup=10, total=100, min_ratio=0.1) == pytest.approx(1.0)
    assert lr_factor(10, warmup=10, total=100, min_ratio=0.1) == pytest.approx(1.0)
    assert lr_factor(100, warmup=10, total=100, min_ratio=0.1) == pytest.approx(0.1)
    assert lr_factor(55, warmup=10, total=100, min_ratio=0.0) == pytest.approx(0.5)


def test_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  lr: 1.0e-4\nmodel:\n  type: lstm\n")
    cfg = load_config(path, ["train.lr=3e-3", "train.max_steps=7", "out_dir=runs/x", "train.amp=false"])
    assert cfg["train"] == {"lr": 3e-3, "max_steps": 7, "amp": False}
    assert cfg["out_dir"] == "runs/x"
    assert _parse_value("lstm") == "lstm"


def test_train_resume_and_finetune(tokens_dir, tmp_path):
    torch.manual_seed(0)
    out = tmp_path / "pre"
    s = train(tiny_cfg(tokens_dir, out))
    assert s["steps"] == 30 and s["best_val_loss"] < 4.0
    assert {"best.pt", "last.pt", "log.csv", "summary.json", "config.yaml"} <= {p.name for p in out.iterdir()}
    with open(out / "log.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [10, 20, 30]

    # retomar continua do passo salvo
    s2 = train(tiny_cfg(tokens_dir, out, max_steps=40), resume=True)
    assert s2["steps"] == 40
    with open(out / "log.csv") as f:
        assert [int(r["step"]) for r in csv.DictReader(f)][-1] == 40

    # ajuste fino parte dos pesos salvos
    s3 = train(tiny_cfg(tokens_dir, tmp_path / "ft", init_from=str(out / "best.pt"), max_steps=10))
    assert s3["best_val_loss"] < 4.0
    model = load_model(tmp_path / "ft" / "best.pt")
    assert model(torch.tensor([[SP.bos, SP.emotions[0]]])).shape == (1, 2, len(TOK))


def test_early_stopping(tokens_dir, tmp_path):
    # lr zero: a validação nunca melhora depois da primeira avaliação
    s = train(tiny_cfg(tokens_dir, tmp_path / "es", lr=0.0, max_steps=100, patience=2))
    assert s["steps"] == 30 and s["best_step"] == 10
