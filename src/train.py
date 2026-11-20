"""Laço de treino (pré-treino e ajuste fino) do Transformer e da LSTM.

Uso: python -m src.train configs/transformer_pretrain.yaml [--set train.max_steps=100] [--resume]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import WindowDataset, collate, load_split, make_loader
from src.model import build_model, count_params, next_token_loss
from src.tokenize import build_tokenizer, special_ids

LOG_FIELDS = ["step", "epoch", "lr", "train_loss", "val_loss", "val_ppl", "elapsed_s"]


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------


def load_config(path: str | Path, overrides: list[str] = ()) -> dict:
    """Lê o YAML e aplica overrides no formato 'secao.chave=valor'."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, _, value = item.partition("=")
        *parents, last = key.split(".")
        node = cfg
        for k in parents:
            node = node.setdefault(k, {})
        node[last] = _parse_value(value)
    return cfg


def _parse_value(value: str):
    # YAML lê "3e-4" (sem ponto) como texto; tenta número antes
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return yaml.safe_load(value)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def amp_settings(device: torch.device, enabled: bool):
    """Contexto de autocast e GradScaler (fp16 precisa de scaler, bf16 não)."""
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext, torch.amp.GradScaler("cuda", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = partial(torch.autocast, device_type="cuda", dtype=dtype)
    return ctx, torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)


def lr_factor(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Warmup linear seguido de decaimento cosseno até min_ratio * lr."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    # sem weight decay em bias e LayerNorm
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def load_model(path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """Carrega um checkpoint salvo por este script (para sample/evaluate)."""
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def save_checkpoint(path: Path, model, opt, sched, scaler, state: dict, model_cfg: dict, cfg: dict):
    # grava num temporário e troca no fim: uma interrupção não corrompe o checkpoint anterior
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "state": state,
        "model_cfg": model_cfg,
        "config": cfg,
    }, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader, pad_id: int, device, amp_ctx) -> float:
    """Perda média por token (sem contar o alvo de emoção nem padding)."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        x = batch["tokens"].to(device)
        with amp_ctx():
            logits = model(x)
        total += next_token_loss(logits.float(), x, pad_id, reduction="sum").item()
        count += int((x[:, 2:] != pad_id).sum())
    model.train(was_training)
    return total / max(count, 1)


def forever(loader):
    """Itera o loader indefinidamente, devolvendo (época, batch)."""
    epoch = 0
    while True:
        for batch in loader:
            yield epoch, batch
        epoch += 1


def build_loaders(cfg: dict, sp, vocab, max_len: int):
    data, tcfg = cfg["data"], cfg["train"]
    train_pieces = load_split(data["dataset"], "train", data.get("tokens_dir", "Data/tokens"))
    val_pieces = load_split(data["dataset"], "val", data.get("tokens_dir", "Data/tokens"))

    n_overfit = tcfg.get("overfit", 0)
    if n_overfit:
        # teste de sanidade: as mesmas N janelas no treino e na validação, sem aumento
        windows = WindowDataset(train_pieces, sp, vocab, max_len, train=False).windows[:n_overfit]
        loader = DataLoader([torch.from_numpy(w) for w in windows], batch_size=n_overfit,
                            collate_fn=partial(collate, pad_id=sp.pad))
        return loader, loader

    common = dict(special=sp, vocab=vocab, batch_size=data["batch_size"], max_len=max_len,
                  num_workers=data.get("num_workers", 0))
    train_loader = make_loader(train_pieces, train=True, cond_dropout=data.get("cond_dropout", 0.15),
                               max_shift=data.get("max_shift", 3), **common)
    val_loader = make_loader(val_pieces, train=False, **common)
    return train_loader, val_loader


def train(cfg: dict, resume: bool = False) -> dict:
    tcfg, data = cfg["train"], cfg["data"]
    seed = tcfg.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = pick_device(tcfg.get("device", "auto"))
    tok = build_tokenizer()
    sp = special_ids(tok)

    model_cfg = {"vocab_size": len(tok), "pad_id": sp.pad, **cfg["model"]}
    max_len = data.get("max_len") or model_cfg.get("max_len", 1024)
    model = build_model(model_cfg).to(device)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    if tcfg.get("init_from") and not resume:
        # ajuste fino: parte dos pesos do pré-treino; otimizador novo
        ckpt = torch.load(tcfg["init_from"], map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"pesos iniciais de {tcfg['init_from']}")

    train_loader, val_loader = build_loaders(cfg, sp, tok.vocab, max_len)

    max_steps = tcfg["max_steps"]
    accum = tcfg.get("grad_accum", 1)
    opt = torch.optim.AdamW(param_groups(model, float(tcfg.get("weight_decay", 0.01))), lr=float(tcfg["lr"]),
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, partial(
        lr_factor, warmup=tcfg.get("warmup_steps", 0), total=max_steps,
        min_ratio=tcfg.get("min_lr_ratio", 0.1)))
    amp_ctx, scaler = amp_settings(device, tcfg.get("amp", True))

    state = {"step": 0, "best_val": float("inf"), "best_step": 0, "bad_evals": 0}
    if resume:
        ckpt = torch.load(out_dir / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        print(f"retomando do passo {state['step']}")

    n_params = count_params(model)
    print(f"{cfg['model']['type']}: {n_params / 1e6:.2f}M parâmetros, dispositivo {device}, "
          f"{len(train_loader)} batches/época")

    log_path = out_dir / "log.csv"
    new_log = not (resume and log_path.exists())
    log_file = open(log_path, "w" if new_log else "a", newline="")
    log = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    if new_log:
        log.writeheader()

    eval_every = tcfg.get("eval_every", 500)
    log_every = tcfg.get("log_every", 50)
    patience = tcfg.get("patience", 0)
    clip = tcfg.get("grad_clip", 1.0)
    batches = forever(train_loader)
    model.train()
    t0 = time.time()
    run_loss, run_n = 0.0, 0

    while state["step"] < max_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            epoch, batch = next(batches)
            x = batch["tokens"].to(device, non_blocking=True)
            with amp_ctx():
                loss = next_token_loss(model(x), x, sp.pad) / accum
            scaler.scale(loss).backward()
            run_loss += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        state["step"] += 1
        run_n += 1
        step = state["step"]

        if step % log_every == 0:
            print(f"passo {step}/{max_steps} época {epoch} perda {run_loss / run_n:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)", flush=True)

        if step % eval_every == 0 or step == max_steps:
            val = evaluate_loss(model, val_loader, sp.pad, device, amp_ctx)
            log.writerow({"step": step, "epoch": epoch, "lr": f"{sched.get_last_lr()[0]:.3e}",
                          "train_loss": f"{run_loss / max(run_n, 1):.5f}", "val_loss": f"{val:.5f}",
                          "val_ppl": f"{math.exp(min(val, 50)):.3f}",
                          "elapsed_s": f"{time.time() - t0:.0f}"})
            log_file.flush()
            run_loss, run_n = 0.0, 0

            improved = val < state["best_val"] - tcfg.get("min_delta", 0.0)
            if improved:
                state.update(best_val=val, best_step=step, bad_evals=0)
                save_checkpoint(out_dir / "best.pt", model, opt, sched, scaler, state, model_cfg, cfg)
            else:
                state["bad_evals"] += 1
            print(f"validação passo {step}: perda {val:.4f} (melhor {state['best_val']:.4f} "
                  f"no passo {state['best_step']}){' *' if improved else ''}", flush=True)
            save_checkpoint(out_dir / "last.pt", model, opt, sched, scaler, state, model_cfg, cfg)

            if patience and state["bad_evals"] >= patience:
                print(f"early stopping: {patience} avaliações sem melhora")
                break

    log_file.close()
    summary = {"best_val_loss": state["best_val"], "best_val_ppl": math.exp(min(state["best_val"], 50)),
               "best_step": state["best_step"], "steps": state["step"], "params": n_params,
               "minutes": round((time.time() - t0) / 60, 1)}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="overrides: train.max_steps=100 ...")
    ap.add_argument("--resume", action="store_true", help="continua de out_dir/last.pt")
    args = ap.parse_args()
    train(load_config(args.config, args.set), resume=args.resume)


if __name__ == "__main__":
    main()
