"""Transformer decoder e LSTM sobre tokens REMI (PyTorch puro)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

PREFIX_LEN = 2  # <BOS> <emoção>


# ---------------------------------------------------------------------------
# Transformer decoder
# ---------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    vocab_size: int
    pad_id: int
    n_layers: int = 6
    d_model: int = 512
    n_heads: int = 8
    ff_mult: int = 4
    max_len: int = 1024
    dropout: float = 0.1
    tie_weights: bool = True


class MusicTransformer(nn.Module):
    supports_state = False

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model precisa ser divisível por n_heads")
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)  # posições aprendidas
        self.drop = nn.Dropout(cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_mult * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pré-LN
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(_init_weights)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

    @property
    def context_len(self) -> int:
        return self.cfg.max_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, T = tokens.shape
        if T > self.cfg.max_len:
            raise ValueError(f"sequência de {T} tokens excede o contexto {self.cfg.max_len}")

        pos = torch.arange(T, device=tokens.device)
        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))

        # máscara causal (True = futuro bloqueado); padding à direita dispensa máscara
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        x = self.blocks(x, mask=causal, is_causal=True)
        return self.head(self.norm(x))


# ---------------------------------------------------------------------------
# LSTM (baseline)
# ---------------------------------------------------------------------------


@dataclass
class LSTMConfig:
    vocab_size: int
    pad_id: int
    n_layers: int = 2
    hidden: int = 512
    d_embed: int = 512
    dropout: float = 0.1
    tie_weights: bool = False


class MusicLSTM(nn.Module):
    supports_state = True  # gera token a token reaproveitando o estado
    context_len = None

    def __init__(self, cfg: LSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_embed)
        self.drop = nn.Dropout(cfg.dropout)
        self.lstm = nn.LSTM(
            input_size=cfg.d_embed,
            hidden_size=cfg.hidden,
            num_layers=cfg.n_layers,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(cfg.hidden, cfg.vocab_size)

        self.emb.weight.data.normal_(0.0, 0.02)
        nn.init.zeros_(self.head.bias)
        if cfg.tie_weights:
            if cfg.hidden != cfg.d_embed:
                raise ValueError("tie_weights exige hidden == d_embed")
            self.head.weight = self.emb.weight

    def forward(
        self,
        tokens: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_state: bool = False,
    ):
        x = self.drop(self.emb(tokens))
        out, state = self.lstm(x, state)
        logits = self.head(self.drop(out))
        return (logits, state) if return_state else logits


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, 0.0, 0.02)


def next_token_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    pad_id: int,
    reduction: str = "mean",
) -> torch.Tensor:
    targets = tokens[:, 1:].clone()
    targets[:, : PREFIX_LEN - 1] = pad_id  # não cobra a previsão do token de emoção
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction=reduction,
    )


@torch.no_grad()
def perplexity(model: nn.Module, batches, pad_id: int, device: str | torch.device) -> float:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for tokens in batches:
        tokens = tokens.to(device)
        logits = model(tokens)
        total += next_token_loss(logits, tokens, pad_id, reduction="sum").item()
        valid = tokens[:, PREFIX_LEN:] != pad_id
        count += int(valid.sum())
    model.train(was_training)
    return float(torch.exp(torch.tensor(total / max(count, 1))))


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Cria o modelo a partir de um dict (ex.: configs/*.yaml) com a chave "type"."""
    cfg = dict(cfg)
    kind = cfg.pop("type")
    classes = {"transformer": (TransformerConfig, MusicTransformer), "lstm": (LSTMConfig, MusicLSTM)}
    if kind not in classes:
        raise ValueError(f"tipo de modelo desconhecido: {kind!r}")
    cfg_cls, model_cls = classes[kind]
    known = {f.name for f in fields(cfg_cls)}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"chaves desconhecidas para {kind}: {sorted(unknown)}")
    return model_cls(cfg_cls(**cfg))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
