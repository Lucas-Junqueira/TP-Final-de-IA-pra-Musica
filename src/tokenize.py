"""Tokenização REMI (miditok) com tokens de emoção, partições e janelas.

Uso: python -m src.tokenize [--limit N] [--roundtrip 5]
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
from miditok import REMI, TokenizerConfig
from miditok.utils import merge_tracks
from symusic import Score

from src.markov import filter_invalid_remi
from src.sample import SpecialIds

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]
SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "NULL", *QUADRANTS]  # PAD fica com id 0

EMOPIA_ROOT = Path("Data/raw/emopia_plus/EMOPIA+")
POP1K7_ROOT = Path("Data/raw/pop1k7/Pop1K7/midi_synchronized")
OUT_DIR = Path("Data/tokens")


# ---------------------------------------------------------------------------
# Tokenizador
# ---------------------------------------------------------------------------


def build_tokenizer() -> REMI:
    cfg = TokenizerConfig(
        pitch_range=(21, 108),  # 88 teclas (limite superior inclusivo)
        beat_res={(0, 4): 4, (4, 12): 2},  # 4 posições por tempo = 16 por compasso 4/4
        num_velocities=16,
        use_tempos=True,
        num_tempos=32,
        tempo_range=(40, 250),
        use_chords=False,
        use_time_signatures=False,  # só 4/4
        use_programs=False,
        use_rests=False,
        use_sustain_pedals=False,
        use_pitchdrum_tokens=False,
        special_tokens=SPECIAL_TOKENS,
    )
    return REMI(cfg)


def special_ids(tokenizer: REMI) -> SpecialIds:
    v = tokenizer.vocab
    return SpecialIds(
        bos=v["BOS_None"],
        eos=v["EOS_None"],
        pad=v["PAD_None"],
        null=v["NULL_None"],
        emotions=[v[f"{q}_None"] for q in QUADRANTS],
    )


# ---------------------------------------------------------------------------
# MIDI <-> sequência
# ---------------------------------------------------------------------------


def load_score(path: str | Path) -> Score | None:
    """Lê o MIDI, junta as trilhas em uma de piano; None se não for 4/4 ou estiver vazio."""
    score = Score(str(path))
    if any((ts.numerator, ts.denominator) != (4, 4) for ts in score.time_signatures):
        return None
    if sum(len(t.notes) for t in score.tracks) == 0:
        return None
    piano = merge_tracks(score.copy())
    piano.program, piano.is_drum, piano.name = 0, False, "piano"
    score.tracks = [piano]
    return score


def encode_events(tokenizer: REMI, score: Score) -> list[int]:
    return list(tokenizer.encode(score)[0].ids)


def to_sequence(events: list[int], cond_id: int, sp: SpecialIds) -> list[int]:
    return [sp.bos, cond_id, *events, sp.eos]


def decode_sequence(tokenizer: REMI, ids: list[int], sp: SpecialIds) -> tuple[Score, int]:
    """Sequência gerada -> Score, descartando tokens fora de ordem; retorna (score, nº descartados)."""
    special = {sp.bos, sp.eos, sp.pad, sp.null, *sp.emotions}
    events = [tokenizer[i] for i in ids if i not in special]
    kept, dropped = filter_invalid_remi(events)
    kept_ids = [tokenizer.vocab[t] for t in kept]
    return tokenizer.decode([kept_ids]), dropped


# ---------------------------------------------------------------------------
# Janelas
# ---------------------------------------------------------------------------


def make_windows(seq: list[int], max_len: int, bar_id: int) -> list[list[int]]:
    """Janelas sem sobreposição de até max_len tokens, com compassos inteiros e <BOS> <emoção> repetidos."""
    prefix, body = seq[:2], seq[2:]
    n = max_len - len(prefix)
    if len(body) <= n:
        return [seq]
    bars = [i for i, t in enumerate(body) if t == bar_id]
    windows, start = [], 0
    while len(body) - start > n:
        # corta no último Bar que cabe; compasso maior que a janela é cortado no meio
        end = max((b for b in bars if start < b <= start + n), default=start + n)
        windows.append(prefix + body[start:end])
        start = end
    windows.append(prefix + body[start:])
    return windows


# ---------------------------------------------------------------------------
# Partições
# ---------------------------------------------------------------------------


def parse_emopia_name(name: str) -> tuple[str, str]:
    """'Q1__8v0MFBZoco_0.mid' -> ('Q1', '_8v0MFBZoco')."""
    stem = Path(name).stem
    return stem[:2], stem[3:].rsplit("_", 1)[0]


def emopia_splits(root: Path) -> dict[str, list[str]]:
    splits = {}
    for split in ("train", "val", "test"):
        with open(root / "split" / f"{split}_clip.csv", newline="") as f:
            splits[split] = [row["clip_name"] for row in csv.DictReader(f)]
    # nenhuma música pode aparecer em mais de uma partição
    songs = {s: {parse_emopia_name(n)[1] for n in names} for s, names in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = songs[a] & songs[b]
        if shared:
            raise ValueError(f"músicas em {a} e {b} ao mesmo tempo: {sorted(shared)[:5]}")
    return splits


def pop1k7_splits(root: Path, val_frac: float = 0.05, seed: int = 0,
                  exclude: set[str] = frozenset()) -> dict[str, list[str]]:
    """Divisão 95/5 por música (cada arquivo é uma música); sem teste."""
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.mid"))
    files = [f for f in files if f not in exclude]  # músicas que também estão no teste do EMOPIA+
    random.Random(seed).shuffle(files)
    n_val = max(1, round(len(files) * val_frac))
    return {"train": sorted(files[n_val:]), "val": sorted(files[:n_val])}


# ---------------------------------------------------------------------------
# Tokenizar e salvar
# ---------------------------------------------------------------------------


def tokenize_split(tokenizer, midi_root: Path, names: list[str], dataset: str, sp: SpecialIds):
    pieces, skipped = [], Counter()
    for name in names:
        try:
            score = load_score(midi_root / name)
        except Exception:
            skipped["erro de leitura"] += 1
            continue
        if score is None:
            skipped["não 4/4 ou vazio"] += 1
            continue
        events = encode_events(tokenizer, score)
        if dataset == "emopia":
            quadrant, song = parse_emopia_name(name)
            cond = sp.emotions[QUADRANTS.index(quadrant)]
        else:
            quadrant, song, cond = None, name.rsplit(".", 1)[0], sp.null
        pieces.append({
            "name": name,
            "song": song,
            "quadrant": quadrant,
            "ids": np.asarray(to_sequence(events, cond, sp), dtype=np.int16),
            "seconds": float(score.to("second").end()),
        })
    return pieces, skipped


def split_stats(pieces: list[dict], skipped: Counter) -> dict:
    lengths = [len(p["ids"]) for p in pieces]
    return {
        "n": len(pieces),
        "por_quadrante": dict(sorted(Counter(p["quadrant"] for p in pieces if p["quadrant"]).items())),
        "duracao_media_s": round(float(np.mean([p["seconds"] for p in pieces])), 2) if pieces else 0,
        "tokens_total": int(sum(lengths)),
        "tokens_medio": round(float(np.mean(lengths)), 1) if pieces else 0,
        "descartados": dict(skipped),
    }


def load_pieces(path: str | Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emopia", type=Path, default=EMOPIA_ROOT)
    ap.add_argument("--pop1k7", type=Path, default=POP1K7_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--exclude", type=Path, default=Path("Data/pop1k7_exclude.txt"),
                    help="peças do Pop1k7 a remover (gerado por python -m src.overlap)")
    ap.add_argument("--limit", type=int, default=0, help="máx. de arquivos por partição (teste rápido)")
    ap.add_argument("--roundtrip", type=int, default=5, help="nº de clipes para o teste de ida e volta")
    args = ap.parse_args()

    tokenizer = build_tokenizer()
    sp = special_ids(tokenizer)
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out / "tokenizer.json")

    exclude = set()
    if args.exclude.exists():
        exclude = {line.strip() for line in args.exclude.read_text().splitlines() if line.strip()}
    print(f"Pop1k7: {len(exclude)} peças excluídas ({args.exclude})")

    stats = {"vocab_size": len(tokenizer), "special_ids": sp.__dict__,
             "pop1k7_excluidas": sorted(exclude)}
    jobs = [("emopia", args.emopia / "midis", emopia_splits(args.emopia)),
            ("pop1k7", args.pop1k7, pop1k7_splits(args.pop1k7, exclude=exclude))]
    for dataset, root, splits in jobs:
        for split, names in splits.items():
            if args.limit:
                names = names[: args.limit]
            pieces, skipped = tokenize_split(tokenizer, root, names, dataset, sp)
            with open(args.out / f"{dataset}_{split}.pkl", "wb") as f:
                pickle.dump(pieces, f)
            stats[f"{dataset}_{split}"] = split_stats(pieces, skipped)
            print(f"{dataset}/{split}: {json.dumps(stats[f'{dataset}_{split}'], ensure_ascii=False)}")

    with open(args.out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"vocabulário: {len(tokenizer)} tokens")

    # teste de ida e volta: MIDI -> tokens -> MIDI, para ouvir/visualizar
    if args.roundtrip:
        rt_dir = Path("samples/roundtrip")
        rt_dir.mkdir(parents=True, exist_ok=True)
        for name in emopia_splits(args.emopia)["train"][: args.roundtrip]:
            score = load_score(args.emopia / "midis" / name)
            seq = to_sequence(encode_events(tokenizer, score), sp.null, sp)
            back, dropped = decode_sequence(tokenizer, seq, sp)
            back.dump_midi(str(rt_dir / name))
            print(f"ida e volta {name}: notas {len(score.tracks[0].notes)} -> "
                  f"{sum(len(t.notes) for t in back.tracks)}, tokens descartados {dropped}")


if __name__ == "__main__":
    main()
