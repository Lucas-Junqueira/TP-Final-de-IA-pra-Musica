"""Etapa 9: aderência à emoção, qualidade (muspy, OA/KL), diversidade, memorização e perplexidade."""

from __future__ import annotations

import json
import math
import tempfile
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde, norm

from src.sample import SpecialIds

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]
METRICS = ["pitch_range", "pitch_class_entropy", "scale_consistency", "polyphony",
           "empty_beat_rate", "groove_consistency"]
RUN_ORDER = ["markov", "lstm", "transformer_scratch_g1"]  # depois os transformer_g* por γ
MEMO_N = 16  # tamanho do n-grama copiado (plano)
DIV_N = 8  # n-grama usado na diversidade


# ---------------------------------------------------------------------------
# Sequências
# ---------------------------------------------------------------------------


def body_of(ids, sp: SpecialIds) -> list[int]:
    """Só os eventos (sem <BOS>, emoção e <EOS>)."""
    return [t for t in ids[2:] if t not in (sp.eos, sp.pad)]


def crop_bars(ids, n_bars: int, bar_id: int, sp: SpecialIds) -> list[int]:
    """Mantém os n_bars primeiros compassos (0 = tudo)."""
    if not n_bars:
        return list(ids)
    out, bars = list(ids[:2]), 0
    for t in ids[2:]:
        if t == bar_id:
            bars += 1
            if bars > n_bars:
                break
        if t == sp.eos:
            break
        out.append(t)
    return out + [sp.eos]


def ngram_hashes(ids, n: int) -> np.ndarray:
    """Hash (uint64, polinomial) de cada n-grama da sequência."""
    a = np.asarray(ids, dtype=np.uint64)
    if len(a) < n:
        return np.empty(0, dtype=np.uint64)
    base = np.uint64(1_000_003)
    powers = np.array([base ** np.uint64(n - 1 - j) for j in range(n)], dtype=np.uint64)
    with np.errstate(over="ignore"):
        return np.lib.stride_tricks.sliding_window_view(a, n) @ powers


# ---------------------------------------------------------------------------
# Qualidade: métricas muspy por amostra
# ---------------------------------------------------------------------------


def muspy_metrics(midi_path: str | Path) -> dict[str, float]:
    import muspy

    music = muspy.read_midi(str(midi_path))
    if not any(len(t.notes) for t in music.tracks):
        return {m: float("nan") for m in METRICS} | {"pch": [0.0] * 12}
    pitches = np.array([n.pitch for t in music.tracks for n in t.notes])
    pch = np.bincount(pitches % 12, minlength=12).astype(float)
    return {
        "pitch_range": float(muspy.pitch_range(music)),
        "pitch_class_entropy": float(muspy.pitch_class_entropy(music)),
        "scale_consistency": float(muspy.scale_consistency(music)),
        "polyphony": float(muspy.polyphony(music)),
        "empty_beat_rate": float(muspy.empty_beat_rate(music)),
        "groove_consistency": float(muspy.groove_consistency(music, 4 * music.resolution)),
        "pch": (pch / pch.sum()).tolist(),  # histograma de classes de altura (diversidade)
    }


def sequences_metrics(seqs, tok, sp, crop: int, bar_id: int) -> list[dict]:
    """Decodifica (com o mesmo filtro REMI) e mede cada sequência."""
    from src.tokenize import decode_sequence

    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, ids in enumerate(seqs):
            score, _ = decode_sequence(tok, crop_bars(ids, crop, bar_id, sp), sp)
            path = Path(tmp) / f"{i}.mid"
            score.dump_midi(str(path))
            out.append(muspy_metrics(path))
    return out


def overlap_and_kl(gen, ref, grid_size: int = 1000) -> tuple[float, float]:
    """Overlap area e KL(ref || gen) entre as KDEs das duas distribuições."""
    gen = np.asarray(gen, float)
    ref = np.asarray(ref, float)
    gen, ref = gen[np.isfinite(gen)], ref[np.isfinite(ref)]
    if len(gen) < 2 or len(ref) < 2:
        return float("nan"), float("nan")
    lo, hi = min(gen.min(), ref.min()), max(gen.max(), ref.max())
    span = max(hi - lo, 1e-6)
    rng = np.random.default_rng(0)
    pdfs = []
    for x in (gen, ref):
        if x.std() < 1e-9 * span + 1e-12:  # valores constantes: KDE singular
            x = x + rng.normal(0, 1e-3 * span, len(x))
        pdfs.append(gaussian_kde(x))
    grid = np.linspace(lo - 0.25 * span, hi + 0.25 * span, grid_size)
    p, q = (np.maximum(k(grid), 0) for k in pdfs)
    p, q = p / trapezoid(p, grid), q / trapezoid(q, grid)
    oa = trapezoid(np.minimum(p, q), grid)
    eps = 1e-10
    kl = trapezoid(q * np.log((q + eps) / (p + eps)), grid)
    return float(oa), float(kl)


def quality_report(gen: list[dict], gen_q: list[str], ref: list[dict], ref_q: list[str]) -> dict:
    """OA/KL e diferença de médias por métrica, comparando quadrante com quadrante."""
    rep = {}
    for m in METRICS:
        oas, kls, diffs = [], [], []
        for q in QUADRANTS:
            g = [d[m] for d, k in zip(gen, gen_q) if k == q]
            r = [d[m] for d, k in zip(ref, ref_q) if k == q]
            oa, kl = overlap_and_kl(g, r)
            oas.append(oa)
            kls.append(kl)
            diffs.append(abs(np.nanmean(g) - np.nanmean(r)))
        rep[m] = {"oa": float(np.nanmean(oas)), "kl": float(np.nanmean(kls)),
                  "mean_diff": float(np.nanmean(diffs)),
                  "mean": float(np.nanmean([d[m] for d in gen])),
                  "oa_by_quadrant": dict(zip(QUADRANTS, oas))}
    rep["oa_mean"] = float(np.nanmean([rep[m]["oa"] for m in METRICS]))
    rep["kl_mean"] = float(np.nanmean([rep[m]["kl"] for m in METRICS]))
    return rep


# ---------------------------------------------------------------------------
# Diversidade e memorização
# ---------------------------------------------------------------------------


def diversity_report(seqs, quadrants, pchs, sp: SpecialIds) -> dict:
    """Média, por quadrante, das distâncias entre pares (histograma de alturas e n-gramas)."""
    pch_d, jac_d = [], []
    for q in QUADRANTS:
        idx = [i for i, k in enumerate(quadrants) if k == q]
        sets = {i: set(ngram_hashes(body_of(seqs[i], sp), DIV_N).tolist()) for i in idx}
        for i, j in combinations(idx, 2):
            pch_d.append(0.5 * np.abs(np.subtract(pchs[i], pchs[j])).sum())  # dist. total variation
            a, b = sets[i], sets[j]
            jac_d.append(1 - len(a & b) / max(len(a | b), 1))
    return {"pch_distance": float(np.mean(pch_d)), "ngram_distance": float(np.mean(jac_d))}


class TrainIndex:
    """Conjunto ordenado de hashes de n-gramas do treino."""

    def __init__(self, sequences, sp: SpecialIds, n: int = MEMO_N):
        self.n = n
        parts = [ngram_hashes(body_of(s, sp), n) for s in sequences]
        self.hashes = np.unique(np.concatenate(parts))

    def matches(self, ids, sp) -> np.ndarray:
        h = ngram_hashes(body_of(ids, sp), self.n)
        if not len(h):
            return np.zeros(0, bool)
        pos = np.searchsorted(self.hashes, h).clip(max=len(self.hashes) - 1)
        return self.hashes[pos] == h


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def memorization_report(seqs, sp, index: TrainIndex) -> dict:
    """Fração de n-gramas copiados e maior trecho copiado (em tokens) por amostra."""
    frac, longest = [], []
    for ids in seqs:
        m = index.matches(ids, sp)
        frac.append(float(m.mean()) if len(m) else 0.0)
        run = longest_run(m)
        longest.append(run + index.n - 1 if run else 0)
    return {"copied_frac": float(np.mean(frac)), "longest_copy_mean": float(np.mean(longest)),
            "longest_copy_max": int(max(longest)),
            "frac_with_copy_ge_64": float(np.mean([x >= 64 for x in longest]))}


# ---------------------------------------------------------------------------
# Atributos musicais (o que o CFG muda)
# ---------------------------------------------------------------------------

# perfis de tonalidade de Krumhansl-Kessler (dó maior / dó menor)
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
ATTRIBUTES = ["notes_per_bar", "velocity", "tempo", "pitch", "duration", "minor"]


def estimate_mode(pch) -> int:
    """1 se a melhor tonalidade (correlação com KK) for menor."""
    pch = np.asarray(pch, float)
    if pch.sum() == 0 or pch.std() == 0:
        return 0
    best = max((np.corrcoef(pch, np.roll(prof, k))[0, 1], is_minor)
               for is_minor, prof in ((0, KK_MAJOR), (1, KK_MINOR)) for k in range(12))
    return best[1]


def attributes(ids, tok, sp: SpecialIds) -> dict[str, float]:
    """Densidade, velocity, andamento, altura, duração média e modo, direto dos tokens."""
    bars, vel, tempo, pitch, dur = 0, [], [], [], []
    pch = np.zeros(12)
    for t in body_of(ids, sp):
        kind, _, value = tok[int(t)].partition("_")
        if kind == "Bar":
            bars += 1
        elif kind == "Pitch":
            pitch.append(int(value))
            pch[int(value) % 12] += 1
        elif kind == "Velocity":
            vel.append(int(value))
        elif kind == "Tempo":
            tempo.append(float(value))
        elif kind == "Duration":
            beats, pos, res = (int(x) for x in value.split("."))
            dur.append(beats + pos / res)
    mean = lambda x: float(np.mean(x)) if x else float("nan")  # noqa: E731
    return {"notes_per_bar": len(pitch) / max(bars, 1), "velocity": mean(vel), "tempo": mean(tempo),
            "pitch": mean(pitch), "duration": mean(dur), "minor": float(estimate_mode(pch))}


def attribute_report(seqs, quadrants, tok, sp) -> dict:
    """Média de cada atributo por quadrante."""
    attrs = [attributes(s, tok, sp) for s in seqs]
    return {q: {a: float(np.nanmean([d[a] for d, k in zip(attrs, quadrants) if k == q])) for a in ATTRIBUTES}
            for q in QUADRANTS}


# ---------------------------------------------------------------------------
# Aderência à emoção
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def two_prop_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """p-valor bilateral do teste z de duas proporções."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return float(2 * norm.sf(abs(k1 / n1 - k2 / n2) / se))


def adherence_report(clf, seqs, quadrants, sp) -> dict:
    from src.classifier import emotion_report, predict

    y_true = [QUADRANTS.index(q) for q in quadrants]
    y_pred = predict(clf, seqs, sp)
    rep = emotion_report(y_true, y_pred)
    n = rep["n"]
    for key in ("acc_4", "acc_valence", "acc_arousal"):
        rep[f"{key}_k"] = round(rep[key] * n)
        rep[f"{key}_ci"] = wilson(rep[f"{key}_k"], n)
    rep["acc_by_quadrant"] = {q: rep["confusion"][i][i] / max(sum(rep["confusion"][i]), 1)
                              for i, q in enumerate(QUADRANTS)}
    return rep


# ---------------------------------------------------------------------------
# Perplexidade no teste
# ---------------------------------------------------------------------------


def test_perplexity(ckpt, pieces, sp, vocab, device, use_null: bool) -> dict:
    from src.dataset import make_loader
    from src.train import evaluate_loss, load_model

    model = load_model(ckpt, device)
    if use_null:  # modelos só pré-treinados nunca viram <Q1..Q4>
        pieces = [dict(p, ids=np.concatenate([[p["ids"][0], sp.null], p["ids"][2:]])) for p in pieces]
    loader = make_loader(pieces, sp, vocab, batch_size=8, train=False, max_len=1024)
    loss = evaluate_loss(model, loader, sp.pad, device, nullcontext)
    del model
    return {"loss": loss, "ppl": math.exp(loss), "condition": "NULL" if use_null else "emoção"}


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

COLORS = {"transformer": "#2a78d6", "lstm": "#eb6834", "markov": "#1baf7a",
          "valence": "#2a78d6", "arousal": "#eb6834", "acc_4": "#6250d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def gamma_of(run: str) -> float | None:
    return float(run.split("_g")[1]) if run.startswith("transformer_g") else None


def fig_tradeoff(results: dict, human: dict, path: Path) -> None:
    """Figura 1: acurácia de emoção × OA médio, com γ anotado."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    _style(ax)
    tr = sorted((gamma_of(r), r) for r in results if gamma_of(r) is not None)
    xs = [results[r]["adherence"]["acc_4"] for _, r in tr]
    ys = [results[r]["quality"]["oa_mean"] for _, r in tr]
    ax.plot(xs, ys, "-o", color=COLORS["transformer"], lw=2, ms=8, label="Transformer (γ)",
            markeredgecolor="white", markeredgewidth=2)
    for k, ((g, _), x, y) in enumerate(zip(tr, xs, ys)):
        dy = 8 if k % 2 == 0 else -14  # alterna acima/abaixo para não sobrepor
        ax.annotate(f"γ={g:g}", (x, y), textcoords="offset points", xytext=(0, dy), ha="center",
                    color=INK, fontsize=9)
    for r, marker in (("lstm", "s"), ("markov", "D")):
        if r in results:
            ax.plot(results[r]["adherence"]["acc_4"], results[r]["quality"]["oa_mean"], marker,
                    color=COLORS[r], ms=9, label=r.upper() if r == "lstm" else "Markov",
                    markeredgecolor="white", markeredgewidth=2)
    ax.axvline(human["adherence"]["acc_4"], color=MUTED, ls="--", lw=1)
    ax.text(human["adherence"]["acc_4"], ax.get_ylim()[0], " teto (teste humano)", color=MUTED,
            fontsize=8, rotation=90, va="bottom", ha="right")
    ax.axvline(0.25, color=MUTED, ls=":", lw=1)
    ax.text(0.25, ax.get_ylim()[0], " acaso", color=MUTED, fontsize=8, rotation=90, va="bottom", ha="right")
    if "train_ref" in human:
        ax.axhline(human["train_ref"]["oa_mean"], color=MUTED, ls="--", lw=1)
        ax.text(ax.get_xlim()[0], human["train_ref"]["oa_mean"], " OA treino humano × teste",
                color=MUTED, fontsize=8, va="bottom")
    ax.set_xlabel("Acurácia de emoção (4 classes, classificador)")
    ax.set_ylabel("OA médio vs. teste humano (qualidade)")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_valence_arousal(results: dict, human: dict, path: Path) -> None:
    """Figura 2: acurácia de valência, arousal e 4 classes em função de γ."""
    import matplotlib.pyplot as plt

    tr = sorted((gamma_of(r), r) for r in results if gamma_of(r) is not None)
    gs = [g for g, _ in tr]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _style(ax)
    labels = {"acc_valence": ("valência", COLORS["valence"]), "acc_arousal": ("arousal", COLORS["arousal"]),
              "acc_4": ("4 classes", COLORS["acc_4"])}
    for key, (name, color) in labels.items():
        acc = [results[r]["adherence"][key] for _, r in tr]
        ci = np.array([results[r]["adherence"][f"{key}_ci"] for _, r in tr])
        ax.errorbar(gs, acc, yerr=[np.subtract(acc, ci[:, 0]), np.subtract(ci[:, 1], acc)], fmt="-o",
                    color=color, lw=2, ms=7, capsize=3, label=name)
        ceil = human["adherence"][key]
        ax.axhline(ceil, color=color, ls="--", lw=1, alpha=0.7)
        ax.text(gs[-1], ceil, f" teto {name}", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(gs)
    ax.set_xlabel("γ (peso do guidance)")
    ax.set_ylabel("Acurácia (classificador), IC 95%")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_loss_curves(runs_dir: Path, path: Path) -> None:
    """Figura 3: perda de treino e validação (pré-treino e ajuste fino)."""
    import csv

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, phase in zip(axes, ("pretrain", "finetune")):
        _style(ax)
        for model in ("transformer", "lstm"):
            log = runs_dir / f"{model}_{phase}" / "log.csv"
            if not log.exists():
                continue
            with open(log) as f:
                rows = list(csv.DictReader(f))
            steps = [int(r["step"]) for r in rows]
            name = "Transformer" if model == "transformer" else "LSTM"
            ax.plot(steps, [float(r["train_loss"]) for r in rows], color=COLORS[model], lw=1.5,
                    alpha=0.5, label=f"{name} treino")
            ax.plot(steps, [float(r["val_loss"]) for r in rows], color=COLORS[model], lw=2,
                    label=f"{name} validação")
        ax.set_title("Pré-treino (Pop1k7)" if phase == "pretrain" else "Ajuste fino (EMOPIA+)", color=INK)
        ax.set_xlabel("passo")
    axes[0].set_ylabel("perda (entropia cruzada por token)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_confusions(results: dict, human: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    items = [("teste humano", human["adherence"])] + [(r, v["adherence"]) for r, v in results.items()]
    cols = 4
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (name, rep) in zip(np.ravel(axes), items):
        ax.axis("on")
        m = np.array(rep["confusion"], float)
        m = m / m.sum(1, keepdims=True).clip(min=1)
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), QUADRANTS, fontsize=8)
        ax.set_yticks(range(4), QUADRANTS, fontsize=8)
        ax.set_title(f"{name} (acc {rep['acc_4']:.2f})", fontsize=9, color=INK)
        ax.set_xlabel("previsto", fontsize=8)
        ax.set_ylabel("pedido", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_metric_oa(results: dict, path: Path) -> None:
    """OA por métrica e por sistema."""
    import matplotlib.pyplot as plt

    runs = list(results)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for ax, m in zip(np.ravel(axes), METRICS):
        _style(ax)
        vals = [results[r]["quality"][m]["oa"] for r in runs]
        colors = [COLORS["transformer"] if r.startswith("transformer") else COLORS.get(r, MUTED) for r in runs]
        ax.bar(range(len(runs)), vals, color=colors, width=0.8)
        ax.set_xticks(range(len(runs)), [r.replace("transformer_", "T ") for r in runs], rotation=45,
                      ha="right", fontsize=8)
        ax.set_title(m, color=INK, fontsize=10)
        ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel("OA vs. teste humano")
    axes[1, 0].set_ylabel("OA vs. teste humano")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


Q_COLORS = dict(zip(QUADRANTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
ATTR_LABELS = {"notes_per_bar": "notas por compasso", "velocity": "velocity média",
               "tempo": "andamento médio (BPM)", "pitch": "altura média (MIDI)",
               "duration": "duração média (tempos)", "minor": "fração em modo menor"}


def fig_attributes(results: dict, human: dict, path: Path) -> None:
    """Atributos por quadrante em função de γ; tracejado = teste humano."""
    import matplotlib.pyplot as plt

    tr = sorted((gamma_of(r), r) for r in results if gamma_of(r) is not None)
    gs = [g for g, _ in tr]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, a in zip(np.ravel(axes), ATTRIBUTES):
        _style(ax)
        for q in QUADRANTS:
            ax.plot(gs, [results[r]["attributes"][q][a] for _, r in tr], "-o", color=Q_COLORS[q], lw=2,
                    ms=5, label=q)
            ax.axhline(human["attributes"][q][a], color=Q_COLORS[q], ls="--", lw=1, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xticks(gs, [f"{g:g}" for g in gs])
        ax.minorticks_off()
        ax.set_title(ATTR_LABELS[a], color=INK, fontsize=10)
        ax.set_xlabel("γ")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               title="quadrante pedido (linha cheia: gerado; tracejada: teste humano)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_pianorolls(records: list[dict], tok, sp, bar_id: int, run: str, path: Path, bars: int = 8) -> None:
    """Análise qualitativa: piano roll da primeira amostra de cada quadrante."""
    import matplotlib.pyplot as plt

    from src.tokenize import decode_sequence

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for ax, q in zip(axes, QUADRANTS):
        rec = next(r for r in records if r["quadrant"] == q)
        score, _ = decode_sequence(tok, crop_bars(rec["ids"], bars, bar_id, sp), sp)
        tpq = score.ticks_per_quarter
        for t in score.tracks:
            for n in t.notes:
                ax.add_patch(plt.Rectangle((n.time / tpq, n.pitch - 0.4), n.duration / tpq, 0.8,
                                           color=COLORS["transformer"], alpha=0.4 + 0.6 * n.velocity / 127))
        ax.set_xlim(0, 4 * bars)
        ax.set_ylim(21, 108)
        for b in range(bars + 1):
            ax.axvline(4 * b, color=GRID, lw=0.8)
        ax.set_ylabel(f"{q} ({rec['name']})\naltura", fontsize=8)
        _style(ax)
    axes[-1].set_xlabel("tempo (semínimas)")
    fig.suptitle(f"{run}: primeiros {bars} compassos", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------


def _fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def write_tables(results: dict, human: dict, ppl: dict, out: Path) -> None:
    import csv

    header = ["sistema", "acc_4", "IC95_acc_4", "p_vs_g1", "acc_valencia", "acc_arousal", "OA_medio",
              "KL_medio", "div_alturas", "div_ngramas", "copiado_16", "maior_copia", "descartados",
              "compassos"]
    rows = []
    for name, r in [("teste humano", human)] + list(results.items()):
        a, q = r.get("adherence", {}), r.get("quality", {})
        d, m, s = r.get("diversity", {}), r.get("memorization", {}), r.get("summary", {})
        ci = a.get("acc_4_ci")
        rows.append([name, _fmt(a.get("acc_4")), f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "—",
                     _fmt(r.get("p_vs_g1"), 4), _fmt(a.get("acc_valence")), _fmt(a.get("acc_arousal")),
                     _fmt(q.get("oa_mean")), _fmt(q.get("kl_mean")), _fmt(d.get("pch_distance")),
                     _fmt(d.get("ngram_distance")), _fmt(m.get("copied_frac")),
                     _fmt(m.get("longest_copy_mean"), 1), _fmt(s.get("drop_rate"), 5),
                     _fmt(s.get("mean_bars"), 1)])
    if "train_ref" in human:
        rows.append(["treino humano (ref. OA)", "—", "—", "—", "—", "—", _fmt(human["train_ref"]["oa_mean"]),
                     _fmt(human["train_ref"]["kl_mean"])] + ["—"] * 6)
    with open(out / "table1.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]

    md += ["", "Acurácia por quadrante pedido:", "", "| sistema | " + " | ".join(QUADRANTS) + " |",
           "|---|" + "---|" * 4]
    for name, r in [("teste humano", human)] + list(results.items()):
        acc = r["adherence"]["acc_by_quadrant"]
        md.append(f"| {name} | " + " | ".join(_fmt(acc[q]) for q in QUADRANTS) + " |")

    md += ["", "OA por métrica:", "", "| sistema | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for name, r in list(results.items()) + ([("treino humano", {"quality": human["train_ref"]})]
                                            if "train_ref" in human else []):
        md.append(f"| {name} | " + " | ".join(_fmt(r["quality"][m]["oa"]) for m in METRICS) + " |")

    md += ["", "Atributos por quadrante pedido (Q1 / Q2 / Q3 / Q4):", "",
           "| sistema | " + " | ".join(ATTR_LABELS[a] for a in ATTRIBUTES) + " |", "|---|" + "---|" * len(ATTRIBUTES)]
    for name, r in [("teste humano", human)] + list(results.items()):
        cells = [" / ".join(_fmt(r["attributes"][q][a], 2 if a == "minor" else 1) for q in QUADRANTS)
                 for a in ATTRIBUTES]
        md.append(f"| {name} | " + " | ".join(cells) + " |")

    if ppl:
        md += ["", "Perplexidade no teste do EMOPIA+:", "", "| modelo | condição | perda | perplexidade |",
               "|---|---|---|---|"]
        for name, p in ppl.items():
            md.append(f"| {name} | {p['condition']} | {p['loss']:.4f} | {p['ppl']:.3f} |")
    (out / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------


def run_sort_key(run: str):
    g = gamma_of(run)
    return (1, g) if g is not None else (0, RUN_ORDER.index(run) if run in RUN_ORDER else 99)


def main() -> None:
    import argparse
    import pickle

    from src.classifier import load_classifier
    from src.dataset import load_split
    from src.tokenize import build_tokenizer, special_ids

    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, default=Path("samples"))
    ap.add_argument("--runs", nargs="*", default=None, help="padrão: todas as pastas com tokens.pkl")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--classifier", default="runs/classifier_quadrant/classifier.pt")
    ap.add_argument("--crop-bars", type=int, default=16,
                    help="compassos usados nas métricas de qualidade/diversidade (teste humano tem ~14)")
    ap.add_argument("--no-memo-pop1k7", action="store_true", help="memorização só contra o EMOPIA+")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--pianoroll-run", default=None, help="padrão: o Transformer com maior acurácia")
    ap.add_argument("--recompute", action="store_true", help="ignora o cache de métricas muspy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = build_tokenizer()
    sp = special_ids(tok)
    bar_id = tok.vocab["Bar_None"]
    fig_dir = args.out / "figures"
    cache_dir = args.out / "cache"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    runs = args.runs or [p.name for p in args.samples.iterdir() if (p / "tokens.pkl").exists()]
    runs = sorted(runs, key=run_sort_key)
    print("execuções:", runs, flush=True)

    def cached_metrics(name, seqs):
        path = cache_dir / f"{name}_crop{args.crop_bars}.json"
        if path.exists() and not args.recompute:
            return json.loads(path.read_text())
        m = sequences_metrics(seqs, tok, sp, args.crop_bars, bar_id)
        path.write_text(json.dumps(m))
        return m

    clf, target = load_classifier(args.classifier, args.device)
    assert target == "quadrant", "use o classificador de 4 classes"

    test = load_split("emopia", "test")
    train = load_split("emopia", "train")
    test_seqs = [p["ids"].tolist() for p in test]
    test_q = [p["quadrant"] for p in test]
    train_seqs = [p["ids"].tolist() for p in train]
    train_q = [p["quadrant"] for p in train]

    print("índices de memorização...", flush=True)
    memo_sets = {"emopia": TrainIndex(train_seqs, sp)}
    if not args.no_memo_pop1k7:
        memo_sets["pop1k7"] = TrainIndex([p["ids"] for p in load_split("pop1k7", "train")], sp)

    def memo(seqs):
        return {k: memorization_report(seqs, sp, idx) for k, idx in memo_sets.items()}

    # referências humanas
    print("teste humano...", flush=True)
    test_m = cached_metrics("human_test", test_seqs)
    train_m = cached_metrics("human_train", train_seqs)
    human = {
        "adherence": adherence_report(clf, test_seqs, test_q, sp),
        "diversity": diversity_report([crop_bars(s, args.crop_bars, bar_id, sp) for s in test_seqs],
                                      test_q, [d["pch"] for d in test_m], sp),
        "train_ref": quality_report(train_m, train_q, test_m, test_q),
        "attributes": attribute_report([crop_bars(s, args.crop_bars, bar_id, sp) for s in test_seqs],
                                       test_q, tok, sp),
    }
    human["memorization_all"] = memo(test_seqs)
    human["memorization"] = human["memorization_all"]["emopia"]

    results = {}
    for run in runs:
        print(f"{run}...", flush=True)
        with open(args.samples / run / "tokens.pkl", "rb") as f:
            records = pickle.load(f)
        seqs = [r["ids"] for r in records]
        quads = [r["quadrant"] for r in records]
        metrics = cached_metrics(run, seqs)
        res = {
            "adherence": adherence_report(clf, seqs, quads, sp),
            "quality": quality_report(metrics, quads, test_m, test_q),
            "diversity": diversity_report([crop_bars(s, args.crop_bars, bar_id, sp) for s in seqs],
                                          quads, [d["pch"] for d in metrics], sp),
            "memorization_all": memo(seqs),
            "attributes": attribute_report([crop_bars(s, args.crop_bars, bar_id, sp) for s in seqs],
                                           quads, tok, sp),
        }
        res["memorization"] = res["memorization_all"]["emopia"]
        summary = args.samples / run / "summary.json"
        if summary.exists():
            res["summary"] = json.loads(summary.read_text())
        results[run] = res
        a = res["adherence"]
        print(f"  acc4 {a['acc_4']:.3f}  val {a['acc_valence']:.3f}  aro {a['acc_arousal']:.3f}  "
              f"OA {res['quality']['oa_mean']:.3f}  div {res['diversity']['pch_distance']:.3f}", flush=True)

    # significância de cada execução contra o Transformer γ = 1
    if "transformer_g1" in results:
        base = results["transformer_g1"]["adherence"]
        for run, r in results.items():
            a = r["adherence"]
            r["p_vs_g1"] = two_prop_p(a["acc_4_k"], a["n"], base["acc_4_k"], base["n"])
            r["p_vs_g1_valence"] = two_prop_p(a["acc_valence_k"], a["n"], base["acc_valence_k"], base["n"])
            r["p_vs_g1_arousal"] = two_prop_p(a["acc_arousal_k"], a["n"], base["acc_arousal_k"], base["n"])

    ppl = {}
    if not args.skip_ppl:
        print("perplexidade no teste...", flush=True)
        for name, use_null in (("transformer_finetune", False), ("transformer_finetune", True),
                               ("transformer_pretrain", True), ("transformer_scratch", False),
                               ("lstm_finetune", False), ("lstm_finetune", True), ("lstm_pretrain", True)):
            ckpt = Path("runs") / name / "best.pt"
            if ckpt.exists():
                key = f"{name} ({'NULL' if use_null else 'emoção'})"
                ppl[key] = test_perplexity(ckpt, test, sp, tok.vocab, args.device, use_null)
                print(f"  {key}: {ppl[key]['ppl']:.3f}", flush=True)

    with open(args.out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"crop_bars": args.crop_bars, "human": human, "systems": results, "perplexity": ppl},
                  f, indent=2, ensure_ascii=False)
    write_tables(results, human, ppl, args.out)

    fig_tradeoff(results, human, fig_dir / "fig1_tradeoff.png")
    fig_valence_arousal(results, human, fig_dir / "fig2_valence_arousal.png")
    fig_loss_curves(Path("runs"), fig_dir / "fig3_loss_curves.png")
    fig_confusions(results, human, fig_dir / "confusion_matrices.png")
    fig_metric_oa(results, fig_dir / "oa_by_metric.png")
    fig_attributes(results, human, fig_dir / "fig4_attributes.png")
    tr = [r for r in results if gamma_of(r) is not None]
    pr_run = args.pianoroll_run or (max(tr, key=lambda r: results[r]["adherence"]["acc_4"]) if tr else runs[0])
    with open(args.samples / pr_run / "tokens.pkl", "rb") as f:
        fig_pianorolls(pickle.load(f), tok, sp, bar_id, pr_run, fig_dir / f"pianoroll_{pr_run}.png")
    print(f"resultados em {args.out}/ (tables.md, table1.csv, metrics.json, figures/)", flush=True)


if __name__ == "__main__":
    main()
