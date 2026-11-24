"""Renderiza amostras em áudio com FluidSynth (Etapa 10).

Escolhe as primeiras amostras de cada quadrante (sem seleção manual) e grava
samples/audio/<sistema>/Q{k}_{i}.<formato>.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]


def find_fluidsynth(path: str | None) -> str:
    for cand in (path, os.environ.get("FLUIDSYNTH"), shutil.which("fluidsynth")):
        if cand and (Path(cand).exists() or shutil.which(cand)):
            return cand
    raise FileNotFoundError("FluidSynth não encontrado: use --fluidsynth ou a variável FLUIDSYNTH")


def default_soundfont() -> str:
    import muspy

    sf = Path(muspy.get_musescore_soundfont_path())
    if not sf.exists():
        muspy.download_musescore_soundfont()
    return str(sf)


def pick(records: list[dict], n: int) -> list[tuple[str, int, list[int]]]:
    """Primeiras n sequências de cada quadrante, na ordem de geração."""
    out, count = [], {q: 0 for q in QUADRANTS}
    for r in records:
        q = r["quadrant"]
        if count[q] < n:
            out.append((q, count[q], list(r["ids"])))
            count[q] += 1
    return out


def render_midi(fs: str, sf: str, midi: Path, out: Path, rate: int, gain: float) -> None:
    cmd = [fs, "-ni", "-q", "-g", str(gain), "-r", str(rate), "-F", str(out), sf, str(midi)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    from src.dataset import load_split
    from src.evaluate import crop_bars
    from src.tokenize import build_tokenizer, decode_sequence, special_ids

    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["transformer_g5", "transformer_g1", "markov", "human"],
                    help="pastas de samples/ ou 'human' (teste do EMOPIA+)")
    ap.add_argument("--n", type=int, default=2, help="amostras por quadrante")
    ap.add_argument("--bars", type=int, default=16, help="compassos renderizados (0 = tudo)")
    ap.add_argument("--format", default="flac", help="flac, wav, ogg...")
    ap.add_argument("--samples", type=Path, default=Path("samples"))
    ap.add_argument("--out", type=Path, default=Path("samples/audio"))
    ap.add_argument("--fluidsynth", default=None, help="caminho do executável (ou variável FLUIDSYNTH)")
    ap.add_argument("--soundfont", default=None, help="padrão: MuseScore General (via muspy)")
    ap.add_argument("--rate", type=int, default=44100)
    ap.add_argument("--gain", type=float, default=0.8)
    args = ap.parse_args()

    fs = find_fluidsynth(args.fluidsynth)
    sf = args.soundfont or default_soundfont()
    tok = build_tokenizer()
    sp = special_ids(tok)
    bar_id = tok.vocab["Bar_None"]

    manifest = []
    with tempfile.TemporaryDirectory() as tmp:
        for system in args.systems:
            if system == "human":
                records = [{"quadrant": p["quadrant"], "ids": p["ids"].tolist(), "name": p["name"]}
                           for p in load_split("emopia", "test")]
            else:
                with open(args.samples / system / "tokens.pkl", "rb") as f:
                    records = pickle.load(f)
            out_dir = args.out / system
            out_dir.mkdir(parents=True, exist_ok=True)

            for q, i, ids in pick(records, args.n):
                if args.bars:
                    ids = crop_bars(ids, args.bars, bar_id, sp)
                score, _ = decode_sequence(tok, ids, sp)
                midi = Path(tmp) / f"{system}_{q}_{i}.mid"
                score.dump_midi(str(midi))
                audio = out_dir / f"{q}_{i:03d}.{args.format}"
                render_midi(fs, sf, midi, audio, args.rate, args.gain)
                shutil.copy(midi, audio.with_suffix(".mid"))
                manifest.append({"system": system, "quadrant": q, "index": i, "file": audio.as_posix()})
                print(audio, flush=True)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} arquivos em {args.out}")


if __name__ == "__main__":
    main()
