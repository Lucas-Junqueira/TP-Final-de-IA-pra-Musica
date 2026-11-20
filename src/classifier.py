"""Classificador de emoção simbólico (Transformer encoder) e métricas de aderência."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.sample import SpecialIds

TARGETS = ("quadrant", "valence", "arousal")
POSITIVE_VALENCE = {0, 3}  # Q1, Q4
HIGH_AROUSAL = {0, 1}  # Q1, Q2


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


@dataclass
class ClassifierConfig:
    vocab_size: int
    pad_id: int
    n_classes: int = 4
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    ff_mult: int = 4
    max_len: int = 512
    dropout: float = 0.2


class EmotionClassifier(nn.Module):
    def __init__(self, cfg: ClassifierConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_mult * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, T = tokens.shape
        if T > self.cfg.max_len:
            raise ValueError(f"janela de {T} tokens excede max_len {self.cfg.max_len}")
        pad = tokens == self.cfg.pad_id
        pos = torch.arange(T, device=tokens.device)
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        x = self.norm(self.encoder(x, src_key_padding_mask=pad))
        # média dos estados, ignorando padding
        keep = (~pad).unsqueeze(-1).to(x.dtype)
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.head(self.drop(pooled))


def save_classifier(model: EmotionClassifier, path, target: str) -> None:
    torch.save({"config": asdict(model.cfg), "target": target, "state_dict": model.state_dict()}, path)


def load_classifier(path, device="cpu") -> tuple[EmotionClassifier, str]:
    ckpt = torch.load(path, map_location=device)
    model = EmotionClassifier(ClassifierConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["target"]


# ---------------------------------------------------------------------------
# Rótulos e janelas
# ---------------------------------------------------------------------------


def quadrant_index(seq: Sequence[int], special: SpecialIds) -> int:
    try:
        return special.emotions.index(seq[1])
    except ValueError:
        raise ValueError(f"sequência sem token de quadrante na posição 1: {seq[:2]}") from None


def to_target(quadrant: int, target: str) -> int:
    if target == "quadrant":
        return quadrant
    if target == "valence":
        return int(quadrant in POSITIVE_VALENCE)
    if target == "arousal":
        return int(quadrant in HIGH_AROUSAL)
    raise ValueError(f"alvo desconhecido: {target!r} (use um de {TARGETS})")


def combine_valence_arousal(valence: Sequence[int], arousal: Sequence[int]) -> list[int]:
    """Previsões binárias -> quadrantes 0..3 (plano B)."""
    table = {(1, 1): 0, (0, 1): 1, (0, 0): 2, (1, 0): 3}
    return [table[(int(v), int(a))] for v, a in zip(valence, arousal)]


def events_of(seq: Sequence[int], special: SpecialIds) -> list[int]:
    # remove <BOS>, o token de emoção (é o rótulo) e o <EOS>
    events = list(seq[2:])
    if events and events[-1] == special.eos:
        events.pop()
    return events


def _random_crop(events: list[int], bos: int, max_len: int, rng: random.Random) -> list[int]:
    n = max_len - 1
    if len(events) <= n:
        return [bos, *events]
    start = rng.randrange(len(events) - n + 1)
    return [bos, *events[start : start + n]]


def _all_windows(events: list[int], bos: int, max_len: int) -> list[list[int]]:
    n = max_len - 1
    if len(events) <= n:
        return [[bos, *events]]
    starts = list(range(0, len(events) - n + 1, n))
    if starts[-1] + n < len(events):
        starts.append(len(events) - n)  # última janela encostada no fim
    return [[bos, *events[s : s + n]] for s in starts]


def _pad_batch(windows: list[list[int]], pad_id: int) -> torch.Tensor:
    T = max(len(w) for w in windows)
    out = torch.full((len(windows), T), pad_id, dtype=torch.long)
    for i, w in enumerate(windows):
        out[i, : len(w)] = torch.tensor(w, dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# Treino e predição
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_proba(
    model: EmotionClassifier,
    sequences: Sequence[Sequence[int]],
    special: SpecialIds,
    batch_size: int = 64,
) -> torch.Tensor:
    """Probabilidades (N, n_classes), média sobre as janelas de cada peça."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    windows, owner = [], []
    for i, seq in enumerate(sequences):
        for w in _all_windows(events_of(seq, special), special.bos, model.cfg.max_len):
            windows.append(w)
            owner.append(i)
    probs = []
    for s in range(0, len(windows), batch_size):
        batch = _pad_batch(windows[s : s + batch_size], special.pad).to(device)
        probs.append(torch.softmax(model(batch).float(), dim=-1).cpu())
    probs = torch.cat(probs)
    owner_t = torch.tensor(owner)
    out = torch.zeros(len(sequences), model.cfg.n_classes)
    out.index_add_(0, owner_t, probs)
    out /= torch.bincount(owner_t, minlength=len(sequences)).clamp(min=1).unsqueeze(1)
    model.train(was_training)
    return out


def predict(model, sequences, special, batch_size: int = 64) -> list[int]:
    return predict_proba(model, sequences, special, batch_size).argmax(-1).tolist()


def train_classifier(
    model: EmotionClassifier,
    train_seqs: Sequence[Sequence[int]],
    val_seqs: Sequence[Sequence[int]],
    special: SpecialIds,
    target: str = "quadrant",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    patience: int = 15,
    augment: Callable[[list[int], random.Random], list[int]] | None = None,
    seed: int = 0,
    log: Callable[[str], None] | None = print,
) -> dict:
    """Treina e mantém os pesos com a melhor acurácia de validação."""
    expected = 4 if target == "quadrant" else 2
    if model.cfg.n_classes != expected:
        raise ValueError(f"alvo {target!r} precisa de n_classes={expected}")
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    rng = random.Random(seed)

    train_events = [events_of(s, special) for s in train_seqs]
    train_y = [to_target(quadrant_index(s, special), target) for s in train_seqs]
    val_y = [to_target(quadrant_index(s, special), target) for s in val_seqs]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_acc": []}
    best_acc, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_events)))
        rng.shuffle(order)
        total, count = 0.0, 0
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            windows = []
            for i in idx:
                ev = augment(train_events[i], rng) if augment else train_events[i]
                windows.append(_random_crop(ev, special.bos, model.cfg.max_len, rng))
            x = _pad_batch(windows, special.pad).to(device)
            y = torch.tensor([train_y[i] for i in idx], device=device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
            count += len(idx)

        pred = predict(model, val_seqs, special)
        acc = sum(p == t for p, t in zip(pred, val_y)) / max(len(val_y), 1)
        history["train_loss"].append(total / count)
        history["val_acc"].append(acc)
        if log:
            log(f"época {epoch + 1}: perda treino {total / count:.4f}, acurácia val {acc:.3f}")

        if acc > best_acc:
            best_acc, best_state, bad_epochs = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:  # early stopping
                break

    model.load_state_dict(best_state)
    model.eval()
    history["best_val_acc"] = best_acc
    return history


# ---------------------------------------------------------------------------
# Métricas de aderência
# ---------------------------------------------------------------------------


def emotion_report(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Acurácia de 4 classes, valência e arousal, e matriz de confusão (linha = verdadeiro)."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true e y_pred precisam ter o mesmo tamanho (> 0)")
    n = len(y_true)
    confusion = [[0] * 4 for _ in range(4)]
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1
    return {
        "n": n,
        "acc_4": sum(t == p for t, p in zip(y_true, y_pred)) / n,
        "acc_valence": sum((t in POSITIVE_VALENCE) == (p in POSITIVE_VALENCE)
                           for t, p in zip(y_true, y_pred)) / n,
        "acc_arousal": sum((t in HIGH_AROUSAL) == (p in HIGH_AROUSAL)
                           for t, p in zip(y_true, y_pred)) / n,
        "confusion": confusion,
    }


def evaluate_sequences(model, sequences, special: SpecialIds) -> dict:
    y_true = [quadrant_index(s, special) for s in sequences]
    return emotion_report(y_true, predict(model, sequences, special))


# ---------------------------------------------------------------------------
# Linha de comando: treina no treino do EMOPIA+, escolhe pela validação, teto no teste
# ---------------------------------------------------------------------------


def _report(model, seqs, special, target) -> dict:
    if target == "quadrant":
        return evaluate_sequences(model, seqs, special)
    y_true = [to_target(quadrant_index(s, special), target) for s in seqs]
    y_pred = predict(model, seqs, special)
    return {"n": len(seqs), f"acc_{target}": sum(t == p for t, p in zip(y_true, y_pred)) / len(seqs)}


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from src.dataset import PitchShifter, load_split
    from src.tokenize import build_tokenizer, special_ids

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=TARGETS, default="quadrant")
    ap.add_argument("--out", type=Path, default=None, help="padrão: runs/classifier_<target>")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.out or Path(f"runs/classifier_{args.target}")
    out.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer()
    sp = special_ids(tok)
    seqs = {s: [p["ids"].tolist() for p in load_split("emopia", s)] for s in ("train", "val", "test")}

    torch.manual_seed(args.seed)
    model = EmotionClassifier(ClassifierConfig(
        vocab_size=len(tok), pad_id=sp.pad, n_classes=4 if args.target == "quadrant" else 2,
        n_layers=args.layers, d_model=args.d_model, max_len=args.max_len)).to(args.device)
    augment = None if args.no_augment else PitchShifter(tok.vocab).random_transpose
    history = train_classifier(model, seqs["train"], seqs["val"], sp, target=args.target,
                               epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                               patience=args.patience, augment=augment, seed=args.seed)
    save_classifier(model, out / "classifier.pt", args.target)

    # acurácia no teste humano = teto da avaliação
    report = {"target": args.target, "val": _report(model, seqs["val"], sp, args.target),
              "test": _report(model, seqs["test"], sp, args.target), "history": history}
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("val", "test")}, indent=2))


if __name__ == "__main__":
    main()
