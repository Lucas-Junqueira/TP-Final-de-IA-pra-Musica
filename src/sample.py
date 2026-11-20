"""Amostragem: temperatura, top-k, top-p e classifier-free guidance."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

NEG_INF = float("-inf")


@dataclass
class SpecialIds:
    bos: int
    eos: int
    pad: int
    null: int
    emotions: list[int] = field(default_factory=list)  # <Q1>..<Q4>

    def banned_during_generation(self) -> list[int]:
        return [self.bos, self.pad, self.null, *self.emotions]


# ---------------------------------------------------------------------------
# Operações sobre logits (B, V)
# ---------------------------------------------------------------------------


def cfg_combine(cond: torch.Tensor, uncond: torch.Tensor, gamma: float) -> torch.Tensor:
    return uncond + gamma * (cond - uncond)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperatura deve ser > 0")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, NEG_INF)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs  # o top-1 sempre fica
    sorted_logits = sorted_logits.masked_fill(mass_before >= p, NEG_INF)
    return torch.full_like(logits, NEG_INF).scatter(-1, order, sorted_logits)


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(
    model: nn.Module,
    emotions: torch.Tensor | list[int],
    special: SpecialIds,
    max_new_tokens: int = 1024,
    gamma: float = 1.0,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
    seed: int | None = None,
    max_bars: int | None = None,
    bar_id: int | None = None,
    use_cache: bool = True,
) -> list[list[int]]:
    """Gera uma sequência [<BOS>, <emoção>, ..., <EOS>] por id de emoção.

    Com max_bars, o Bar que abriria o compasso max_bars + 1 vira <EOS>.
    Com cache (Transformer), ao encher o contexto refaz o prefill com
    <BOS> <emoção> + os últimos context_len // 2 tokens.
    """
    device = next(model.parameters()).device
    emotions = torch.as_tensor(emotions, dtype=torch.long, device=device)
    B = emotions.numel()
    use_cfg = gamma != 1.0  # gamma = 1 é condicionamento simples

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    bos = torch.full((B,), special.bos, dtype=torch.long, device=device)
    prefix = torch.stack([bos, emotions], dim=1)
    if use_cfg:
        # linhas B..2B: mesma história com <NULL>, no mesmo batch
        null = torch.full((B,), special.null, dtype=torch.long, device=device)
        prefix = torch.cat([prefix, torch.stack([bos, null], dim=1)], dim=0)
    n_rows = prefix.size(0)

    banned = torch.tensor(special.banned_during_generation(), dtype=torch.long, device=device)
    body = torch.empty((B, 0), dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    n_bars = torch.zeros(B, dtype=torch.long, device=device)

    was_training = model.training
    model.eval()

    stateful = getattr(model, "supports_state", False)
    use_kv = use_cache and not stateful and hasattr(model, "step")
    state = None
    if stateful:
        out, state = model(prefix, return_state=True)
        last_logits = out[:, -1]
    if use_kv:
        out, cache = model.step(prefix)
        cache_len = prefix.size(1)
        keep = model.context_len // 2
        last_logits = out[:, -1]

    for _ in range(max_new_tokens):
        if not stateful and not use_kv:
            # janela: <BOS> <emoção> + últimos tokens gerados
            ctx = model.context_len - prefix.size(1)
            hist = body[:, -ctx:] if body.size(1) > ctx else body
            if use_cfg:
                hist = hist.repeat(2, 1)
            last_logits = model(torch.cat([prefix, hist], dim=1))[:, -1]

        logits = last_logits.float()
        if use_cfg:
            logits = cfg_combine(logits[:B], logits[B:], gamma)
        logits[:, banned] = NEG_INF

        nxt = sample_next(logits, temperature, top_k, top_p, generator)
        if max_bars is not None:
            is_bar = nxt == bar_id
            over = is_bar & (n_bars >= max_bars)
            nxt = torch.where(over, torch.full_like(nxt, special.eos), nxt)
            n_bars += (is_bar & ~over).long()
        nxt = torch.where(finished, torch.full_like(nxt, special.pad), nxt)
        finished |= nxt == special.eos
        body = torch.cat([body, nxt[:, None]], dim=1)

        if finished.all():
            break
        if stateful:
            step = nxt.repeat(n_rows // B)[:, None]
            out, state = model(step, state=state, return_state=True)
            last_logits = out[:, -1]
        elif use_kv:
            if cache_len + 1 > model.context_len:
                # contexto cheio: novo prefill com o prefixo + metade final do contexto
                hist = body[:, -keep:].repeat(n_rows // B, 1)
                inp = torch.cat([prefix, hist], dim=1)
                out, cache = model.step(inp, cache, 0)  # reaproveita os buffers
                cache_len = inp.size(1)
            else:
                out, cache = model.step(nxt.repeat(n_rows // B)[:, None], cache, cache_len)
                cache_len += 1
            last_logits = out[:, -1]

    model.train(was_training)

    results = []
    for i in range(B):
        seq = prefix[i].tolist()
        for tok in body[i].tolist():
            if tok == special.pad:
                break
            seq.append(tok)
            if tok == special.eos:
                break
        results.append(seq)
    return results


# ---------------------------------------------------------------------------
# Linha de comando (Etapa 7): N amostras por quadrante para cada sistema e γ
# Uso: python -m src.sample --systems markov lstm transformer --n 100
# ---------------------------------------------------------------------------

GAMMAS = [1.0, 1.5, 2.0, 3.0, 5.0]
SYSTEMS = {  # nome -> (checkpoint, valores de γ)
    "markov": ("runs/markov/markov.pkl", [1.0]),
    "lstm": ("runs/lstm_finetune/best.pt", [1.0]),
    "transformer": ("runs/transformer_finetune/best.pt", GAMMAS),
    "transformer_scratch": ("runs/transformer_scratch/best.pt", [1.0]),
}


def run_name(system: str, gamma: float) -> str:
    return f"{system}_g{gamma:g}" if system.startswith("transformer") else system


def main() -> None:
    import argparse
    import json
    import pickle
    import time
    from pathlib import Path

    from src.markov import QuadrantMarkov
    from src.tokenize import QUADRANTS, build_tokenizer, decode_sequence, special_ids
    from src.train import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS)[:3], choices=list(SYSTEMS))
    ap.add_argument("--gammas", nargs="+", type=float, default=None, help="padrão: os do plano")
    ap.add_argument("--checkpoint", default=None, help="sobrescreve o checkpoint (um sistema)")
    ap.add_argument("--n", type=int, default=100, help="amostras por quadrante")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-bars", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=4096)  # ~85 tokens/compasso no EMOPIA+
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("samples"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tok = build_tokenizer()
    sp = special_ids(tok)
    bar_id = tok.vocab["Bar_None"]

    for system in args.systems:
        ckpt, gammas = SYSTEMS[system]
        ckpt = args.checkpoint or ckpt
        if system == "markov":
            model = QuadrantMarkov.load(ckpt)
        else:
            model = load_model(ckpt, args.device)
        for gamma in (args.gammas if args.gammas and system.startswith("transformer") else gammas):
            out = args.out / run_name(system, gamma)
            if (out / "tokens.pkl").exists() and not args.overwrite:
                print(f"{out} já existe, pulando (use --overwrite)")
                continue
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            records = []
            for qi, emo in enumerate(sp.emotions):
                for start in range(0, args.n, args.batch_size):
                    size = min(args.batch_size, args.n - start)
                    seed = args.seed + 1000 * qi + start  # mesma semente para todos os sistemas
                    if system == "markov":
                        seqs = [model.generate(emo, args.max_new_tokens, args.temperature,
                                               top_p=args.top_p, seed=seed + i,
                                               max_bars=args.max_bars, bar_id=bar_id)
                                for i in range(size)]
                    else:
                        seqs = generate(model, [emo] * size, sp, args.max_new_tokens, gamma,
                                        args.temperature, 0, args.top_p, seed,
                                        max_bars=args.max_bars, bar_id=bar_id)
                    for i, seq in enumerate(seqs):
                        name = f"{QUADRANTS[qi]}_{start + i:03d}"
                        score, dropped = decode_sequence(tok, seq, sp)
                        score.dump_midi(str(out / f"{name}.mid"))
                        records.append({"name": name, "quadrant": QUADRANTS[qi], "ids": seq,
                                        "dropped": dropped, "bars": seq.count(bar_id),
                                        "eos": seq[-1] == sp.eos})
                print(f"{out.name} {QUADRANTS[qi]}: {len(records)} amostras ({time.time() - t0:.0f}s)",
                      flush=True)

            n_events = sum(len(r["ids"]) - 2 for r in records)
            summary = {
                "system": system, "gamma": gamma, "n": len(records),
                "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
                "max_bars": args.max_bars,
                "drop_rate": sum(r["dropped"] for r in records) / max(n_events, 1),
                "mean_tokens": n_events / len(records),
                "mean_bars": sum(r["bars"] for r in records) / len(records),
                "eos_rate": sum(r["eos"] for r in records) / len(records),
                "minutes": round((time.time() - t0) / 60, 1),
            }
            with open(out / "tokens.pkl", "wb") as f:
                pickle.dump(records, f)
            with open(out / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
