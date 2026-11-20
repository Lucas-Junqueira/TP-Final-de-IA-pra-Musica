"""Dataset de janelas REMI: recorte, transposição, dropout de condição e batches com padding."""

from __future__ import annotations

import math
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.sample import SpecialIds
from src.tokenize import OUT_DIR, load_pieces, make_windows


class PitchShifter:
    """Transpõe tokens Pitch; os demais tokens ficam iguais."""

    def __init__(self, vocab: dict[str, int], max_shift: int = 3):
        self.max_shift = max_shift
        size = max(vocab.values()) + 1
        self.pitch_of = np.full(size, -1, dtype=np.int64)  # id -> altura MIDI (-1 se não é Pitch)
        by_pitch = {}
        for tok, i in vocab.items():
            if tok.startswith("Pitch_"):
                p = int(tok.split("_")[1])
                self.pitch_of[i] = p
                by_pitch[p] = i
        self.low, self.high = min(by_pitch), max(by_pitch)
        # tabelas id -> id para cada deslocamento
        self.tables = {}
        for s in range(-max_shift, max_shift + 1):
            table = np.arange(size, dtype=np.int64)
            for p, i in by_pitch.items():
                if p + s in by_pitch:
                    table[i] = by_pitch[p + s]
            self.tables[s] = table

    def shift(self, ids, s: int) -> np.ndarray:
        return self.tables[s][np.asarray(ids, dtype=np.int64)]

    def allowed_shifts(self, ids) -> list[int]:
        """Deslocamentos que mantêm todas as notas dentro da faixa do piano."""
        pitches = self.pitch_of[np.asarray(ids, dtype=np.int64)]
        pitches = pitches[pitches >= 0]
        if pitches.size == 0:
            return [0]
        lo, hi = int(pitches.min()), int(pitches.max())
        return [s for s in self.tables if self.low <= lo + s and hi + s <= self.high]

    def random_transpose(self, ids, rng: random.Random) -> list[int]:
        """Assinatura compatível com o augment de classifier.train_classifier."""
        return self.shift(ids, rng.choice(self.allowed_shifts(ids))).tolist()


class WindowDataset(Dataset):
    """Janelas de até max_len tokens, sempre começando com <BOS> <emoção> Bar.

    train=True: janela aleatória (início em Bar), transposição e dropout de condição.
    train=False: janelas fixas de make_windows, sem aumento (validação).
    """

    def __init__(
        self,
        pieces: list[dict],
        special: SpecialIds,
        vocab: dict[str, int],
        max_len: int = 1024,
        train: bool = True,
        max_shift: int = 3,
        cond_dropout: float = 0.15,
    ):
        self.sp = special
        self.max_len = max_len
        self.train = train
        self.cond_dropout = cond_dropout
        self.bar_id = vocab["Bar_None"]
        self.shifter = PitchShifter(vocab, max_shift) if train and max_shift > 0 else None
        self.pieces = [np.asarray(p["ids"], dtype=np.int64) for p in pieces]
        body_len = max_len - 2

        if train:
            # uma amostra por janela que a peça renderia; o recorte é sorteado a cada acesso
            self.index = []
            self.starts = []
            for k, ids in enumerate(self.pieces):
                body = ids[2:]
                bars = np.flatnonzero(body == self.bar_id)
                last = len(body) - body_len
                starts = bars[bars <= max(last, 0)].tolist()
                later = bars[bars > max(last, 0)]
                if later.size and last > 0:
                    starts.append(int(later[0]))  # garante janelas que chegam ao <EOS>
                self.starts.append(starts or [0])
                self.index += [k] * max(1, math.ceil(len(body) / body_len))
        else:
            self.windows = [
                np.asarray(w, dtype=np.int64)
                for ids in self.pieces
                for w in make_windows(ids.tolist(), max_len, self.bar_id)
            ]

    def __len__(self) -> int:
        return len(self.index) if self.train else len(self.windows)

    def __getitem__(self, i: int) -> torch.Tensor:
        if not self.train:
            return torch.from_numpy(self.windows[i])

        rng = random.Random(int(torch.randint(2**31, (1,))))  # segue a semente do torch/worker
        ids = self.pieces[self.index[i]]
        start = rng.choice(self.starts[self.index[i]])
        body = ids[2 + start : 2 + start + self.max_len - 2]
        if self.shifter is not None:
            body = self.shifter.shift(body, rng.choice(self.shifter.allowed_shifts(body)))
        cond = int(ids[1])
        if cond != self.sp.null and rng.random() < self.cond_dropout:
            cond = self.sp.null  # dropout de condição (necessário para o CFG)
        return torch.from_numpy(np.concatenate([[self.sp.bos, cond], body]).astype(np.int64))


def collate(batch: list[torch.Tensor], pad_id: int) -> dict[str, torch.Tensor]:
    """Padding à direita; mask = True nos tokens reais."""
    T = max(len(x) for x in batch)
    tokens = torch.full((len(batch), T), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        tokens[i, : len(x)] = x
    return {"tokens": tokens, "mask": tokens != pad_id}


def load_split(dataset: str, split: str, tokens_dir: str | Path = OUT_DIR) -> list[dict]:
    if dataset == "pop1k7" and split == "test":
        raise ValueError("Pop1k7 não tem teste")
    return load_pieces(Path(tokens_dir) / f"{dataset}_{split}.pkl")


def make_loader(
    pieces: list[dict],
    special: SpecialIds,
    vocab: dict[str, int],
    batch_size: int,
    train: bool,
    max_len: int = 1024,
    cond_dropout: float = 0.15,
    max_shift: int = 3,
    num_workers: int = 0,
) -> DataLoader:
    ds = WindowDataset(pieces, special, vocab, max_len, train, max_shift, cond_dropout)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=partial(collate, pad_id=special.pad),  # partial: picklável para workers no Windows
        drop_last=False,
    )
