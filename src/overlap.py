"""Sobreposição de músicas entre EMOPIA+ e Pop1k7, por conteúdo (os IDs não batem).

Para cada clipe do EMOPIA+, mede a fração das suas n-gramas que aparece em cada peça do Pop1k7:
  * "gravacao": n-gramas de acordes (notas com início a até 50 ms) -> mesma gravação/transcrição;
  * "melodia": n-gramas de intervalos da nota mais aguda -> mesma música, mesmo em outro arranjo/tom.

Uso: python -m src.overlap   (grava Data/overlap.json e Data/pop1k7_exclude.txt)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from symusic import Score

from src.tokenize import EMOPIA_ROOT, POP1K7_ROOT, emopia_splits, parse_emopia_name

CHORD_WINDOW_S = 0.05
N_CHORD = 6
N_MELODY = 12


def chord_sequence(path: Path) -> list[tuple[int, ...]]:
    """Acordes em ordem de início (em segundos), cada um como tupla de alturas ordenadas."""
    score = Score(str(path)).to("second")
    notes = sorted((n.time, n.pitch) for t in score.tracks for n in t.notes)
    chords, current, start = [], [], None
    for time, pitch in notes:
        if start is None or time - start > CHORD_WINDOW_S:
            if current:
                chords.append(tuple(sorted(set(current))))
            current, start = [], time
        current.append(pitch)
    if current:
        chords.append(tuple(sorted(set(current))))
    return chords


def fingerprints(chords: list[tuple[int, ...]]) -> dict[str, set[int]]:
    top = [c[-1] for c in chords]
    intervals = [b - a for a, b in zip(top, top[1:])]
    return {
        "gravacao": {hash(tuple(chords[i : i + N_CHORD])) for i in range(len(chords) - N_CHORD + 1)},
        "melodia": {hash(tuple(intervals[i : i + N_MELODY])) for i in range(len(intervals) - N_MELODY + 1)},
    }


def find_overlaps(emopia_root: Path, pop_root: Path, min_score: float = 0.1) -> list[dict]:
    """Pares (clipe, peça) com pontuação >= min_score em alguma das impressões digitais."""
    splits = emopia_splits(emopia_root)
    clips = [(split, name) for split, names in splits.items() for name in names]

    # índice invertido: n-grama -> clipes que a contêm
    index = {kind: defaultdict(list) for kind in ("gravacao", "melodia")}
    sizes = {}
    for ci, (_, name) in enumerate(clips):
        fp = fingerprints(chord_sequence(emopia_root / "midis" / name))
        for kind, grams in fp.items():
            sizes[ci, kind] = len(grams)
            for g in grams:
                index[kind][g].append(ci)

    pop_files = sorted(pop_root.rglob("*.mid"))
    hits = defaultdict(int)  # (clipe, peça, tipo) -> nº de n-gramas em comum
    for pi, path in enumerate(pop_files):
        fp = fingerprints(chord_sequence(path))
        for kind, grams in fp.items():
            for g in grams & index[kind].keys():
                for ci in index[kind][g]:
                    hits[ci, pi, kind] += 1
        if (pi + 1) % 250 == 0:
            print(f"{pi + 1}/{len(pop_files)} peças do Pop1k7", flush=True)

    pairs = defaultdict(dict)
    for (ci, pi, kind), n in hits.items():
        pairs[ci, pi][kind] = n / max(sizes[ci, kind], 1)
    results = []
    for (ci, pi), scores in pairs.items():
        if max(scores.values()) >= min_score:
            split, name = clips[ci]
            results.append({
                "emopia_split": split,
                "emopia_clip": name,
                "emopia_song": parse_emopia_name(name)[1],
                "pop1k7": pop_files[pi].relative_to(pop_root).as_posix(),
                "gravacao": round(scores.get("gravacao", 0.0), 3),
                "melodia": round(scores.get("melodia", 0.0), 3),
            })
    return sorted(results, key=lambda r: (-r["gravacao"], -r["melodia"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emopia", type=Path, default=EMOPIA_ROOT)
    ap.add_argument("--pop1k7", type=Path, default=POP1K7_ROOT)
    # pares aleatórios ficam perto de 0 nas duas pontuações; 0.1 já é bem acima do fundo
    ap.add_argument("--threshold", type=float, default=0.1, help="pontuação para considerar a mesma música")
    ap.add_argument("--out", type=Path, default=Path("Data"))
    args = ap.parse_args()

    results = find_overlaps(args.emopia, args.pop1k7)
    with open(args.out / "overlap.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    same = [r for r in results if max(r["gravacao"], r["melodia"]) >= args.threshold]
    exclude = sorted({r["pop1k7"] for r in same if r["emopia_split"] == "test"})
    with open(args.out / "pop1k7_exclude.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{p}\n" for p in exclude)

    print(f"pares acima de {args.threshold}: {len(same)}")
    for r in results[:25]:
        print(f"  {r['emopia_split']:5s} {r['emopia_clip']:28s} {r['pop1k7']:16s} "
              f"gravação {r['gravacao']:.2f} melodia {r['melodia']:.2f}")
    print(f"peças do Pop1k7 a remover (sobrepõem o teste): {len(exclude)} -> {args.out / 'pop1k7_exclude.txt'}")


if __name__ == "__main__":
    main()
