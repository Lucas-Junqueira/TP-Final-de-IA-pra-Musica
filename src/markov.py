"""Baseline de Markov: cadeia de ordem n sobre tokens REMI, uma por quadrante."""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from src.sample import SpecialIds, sample_next


class QuadrantMarkov:
    def __init__(self, order: int, special: SpecialIds):
        if order < 1:
            raise ValueError(f"a ordem deve ser >= 1, recebi {order}")
        if not special.emotions:
            raise ValueError("special.emotions precisa listar os ids de <Q1>..<Q4>")
        self.order = order
        self.special = special
        # counts[emoção][contexto] = Counter(próximo token)
        self.counts: dict[int, dict[tuple[int, ...], Counter]] = {}

    def fit(self, sequences: Iterable[Sequence[int]]) -> "QuadrantMarkov":
        """Conta n-gramas de ordem 0..order por quadrante."""
        sp = self.special
        self.counts = {e: defaultdict(Counter) for e in sp.emotions}
        for seq in sequences:
            if len(seq) < 2 or seq[0] != sp.bos or seq[1] not in self.counts:
                continue  # sem quadrante (ex.: <NULL>)
            events = list(seq[2:])
            if not events or events[-1] != sp.eos:
                events.append(sp.eos)
            table = self.counts[seq[1]]
            history = [sp.bos, *events]
            for i in range(1, len(history)):
                nxt = history[i]
                for k in range(0, min(self.order, i) + 1):
                    table[tuple(history[i - k : i])][nxt] += 1
        for e, table in self.counts.items():
            if not table:
                raise ValueError(f"nenhuma sequência de treino para a emoção {e}")
            self.counts[e] = dict(table)
        return self

    def next_counts(self, emotion: int, history: Sequence[int]) -> Counter:
        """Contagens do maior contexto já visto (fallback até a ordem 0)."""
        table = self.counts[emotion]
        for k in range(min(self.order, len(history)), -1, -1):
            ctx = tuple(history[len(history) - k :])
            if ctx in table:
                return table[ctx]
        raise RuntimeError("modelo vazio: chame fit() antes de generate()")

    def generate(
        self,
        emotion: int,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        max_bars: int | None = None,
        bar_id: int | None = None,
    ) -> list[int]:
        if emotion not in self.counts:
            raise ValueError(f"emoção {emotion} desconhecida ou modelo não treinado")
        sp = self.special
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        history = [sp.bos]
        n_bars = 0
        for _ in range(max_new_tokens):
            counts = self.next_counts(emotion, history)
            options = list(counts.keys())
            # log(contagem) como logit: mesma amostragem dos modelos neurais
            logits = torch.tensor([list(counts.values())], dtype=torch.float).log()
            nxt = options[sample_next(logits, temperature, top_k, top_p, generator).item()]
            if max_bars is not None and nxt == bar_id:
                if n_bars >= max_bars:
                    nxt = sp.eos  # limite de compassos
                else:
                    n_bars += 1
            history.append(nxt)
            if nxt == sp.eos:
                break
        return [sp.bos, emotion, *history[1:]]

    def save(self, path: str | Path) -> None:
        # só dados (sem a classe), para carregar de qualquer módulo
        sp = self.special
        state = {"order": self.order, "special": [sp.bos, sp.eos, sp.pad, sp.null, list(sp.emotions)],
                 "counts": {e: {ctx: dict(c) for ctx, c in t.items()} for e, t in self.counts.items()}}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(path: str | Path) -> "QuadrantMarkov":
        with open(path, "rb") as f:
            state = pickle.load(f)
        bos, eos, pad, null, emotions = state["special"]
        model = QuadrantMarkov(state["order"], SpecialIds(bos, eos, pad, null, emotions))
        model.counts = {e: {ctx: Counter(c) for ctx, c in t.items()} for e, t in state["counts"].items()}
        return model


def main() -> None:
    """Treina no treino do EMOPIA+ (peças inteiras) e salva o modelo."""
    import argparse

    from src.dataset import load_split
    from src.tokenize import build_tokenizer, special_ids

    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/markov/markov.pkl"))
    args = ap.parse_args()

    tok = build_tokenizer()
    sp = special_ids(tok)
    seqs = [p["ids"].tolist() for p in load_split("emopia", "train")]
    model = QuadrantMarkov(args.order, sp).fit(seqs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    for q, e in enumerate(sp.emotions, 1):
        table = model.counts[e]
        n_ctx = sum(len(ctx) == args.order for ctx in table)
        print(f"Q{q}: {n_ctx} contextos de ordem {args.order}")
    print(f"salvo em {args.out}")


def filter_invalid_remi(tokens: Sequence[str]) -> tuple[list[str], int]:
    """Descarta eventos REMI fora de ordem; retorna (mantidos, nº descartados)."""
    kept: list[str] = []
    dropped = 0
    seen_bar = False
    has_position = False
    last_position = -1
    note: list[str] = []  # trinca Pitch -> Velocity -> Duration em construção
    prev_type = None

    for tok in tokens:
        kind, _, value = tok.partition("_")

        if note:
            expected = "Velocity" if len(note) == 1 else "Duration"
            if kind == expected:
                note.append(tok)
                if kind == "Duration":
                    kept.extend(note)
                    note.clear()
                    prev_type = "Duration"
                continue
            dropped += len(note)  # trinca incompleta
            note.clear()

        ok = False
        if kind == "Bar":
            ok = True
            seen_bar, has_position, last_position = True, False, -1
        elif kind == "TimeSig":
            ok = seen_bar and prev_type == "Bar"
        elif kind == "Position" and seen_bar:
            try:
                pos = int(value)
            except ValueError:
                pos = -1
            if pos > last_position:  # posição só avança dentro do compasso
                ok, has_position, last_position = True, True, pos
        elif kind == "Tempo":
            ok = has_position and prev_type in ("Position", "Duration")
        elif kind == "Pitch" and has_position:
            note.append(tok)
            continue

        if ok:
            kept.append(tok)
            prev_type = kind
        else:
            dropped += 1

    dropped += len(note)
    return kept, dropped


if __name__ == "__main__":
    main()
