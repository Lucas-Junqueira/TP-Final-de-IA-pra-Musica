"""Testes de sanidade de src/model.py e src/sample.py (rodar: python -m pytest tests)."""

import pytest
import torch

from src.model import (
    LSTMConfig,
    MusicLSTM,
    MusicTransformer,
    TransformerConfig,
    build_model,
    count_params,
    next_token_loss,
)
from src.sample import SpecialIds, cfg_combine, generate, top_k_filter, top_p_filter

# Vocabulário de brinquedo: 0=PAD 1=BOS 2=EOS 3=NULL 4..7=Q1..Q4, resto = eventos.
V = 40
SPECIAL = SpecialIds(bos=1, eos=2, pad=0, null=3, emotions=[4, 5, 6, 7])


def tiny_transformer(max_len=64, dropout=0.0):
    return MusicTransformer(
        TransformerConfig(vocab_size=V, pad_id=0, n_layers=2, d_model=64, n_heads=4,
                          max_len=max_len, dropout=dropout)
    )


def tiny_lstm(dropout=0.0):
    return MusicLSTM(LSTMConfig(vocab_size=V, pad_id=0, n_layers=2, hidden=64, d_embed=64,
                                dropout=dropout))


def random_batch(B=10, T=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    body = torch.randint(8, V, (B, T - 2), generator=g)
    emo = torch.randint(4, 8, (B, 1), generator=g)
    return torch.cat([torch.full((B, 1), 1), emo, body], dim=1)


@pytest.mark.parametrize("make", [tiny_transformer, tiny_lstm])
def test_output_shape(make):
    x = random_batch(B=3, T=20)
    assert make()(x).shape == (3, 20, V)


def test_transformer_is_causal():
    torch.manual_seed(0)
    model = tiny_transformer().eval()
    x = random_batch(B=2, T=30)
    y = x.clone()
    y[:, 20:] = 8  # muda só o futuro
    with torch.no_grad():
        assert torch.allclose(model(x)[:, :20], model(y)[:, :20], atol=1e-5)
        assert not torch.allclose(model(x)[:, 20:], model(y)[:, 20:])


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = tiny_transformer().eval()
    x = random_batch(B=3, T=40)
    with torch.no_grad():
        full = model(x)
        out, cache = model.step(x[:, :10])  # prefill
        steps = [out]
        out, cache = model.step(x[:, 10:13], cache, 10)  # vários tokens com cache
        steps.append(out)
        for t in range(13, 40):
            out, cache = model.step(x[:, t : t + 1], cache, t)
            steps.append(out)
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)


@pytest.mark.parametrize("gamma", [1.0, 3.0])
def test_generate_with_cache_matches_recompute(gamma):
    torch.manual_seed(0)
    model = tiny_transformer(max_len=64)
    kw = dict(max_new_tokens=50, gamma=gamma, top_p=0.9, seed=3)
    assert generate(model, [4, 6], SPECIAL, use_cache=True, **kw) == \
        generate(model, [4, 6], SPECIAL, use_cache=False, **kw)
    # além do contexto: o cache refaz o prefill e continua gerando
    long = generate(model, [4], SPECIAL, max_new_tokens=150, gamma=gamma, seed=3)
    assert len(long[0]) > 64


def test_transformer_rejects_too_long_input():
    with pytest.raises(ValueError):
        tiny_transformer(max_len=16)(random_batch(B=1, T=17))


def test_lstm_incremental_matches_full():
    torch.manual_seed(0)
    model = tiny_lstm().eval()
    x = random_batch(B=2, T=15)
    with torch.no_grad():
        full = model(x)
        out, state = model(x[:, :5], return_state=True)
        steps = [out]
        for t in range(5, 15):
            out, state = model(x[:, t : t + 1], state=state, return_state=True)
            steps.append(out)
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_loss_ignores_pad_and_emotion_target():
    x = random_batch(B=2, T=10)
    logits = torch.randn(2, 10, V)
    base = next_token_loss(logits, x, pad_id=0)
    # Mudar o logit que prevê o token de emoção (posição 0) não altera a perda.
    logits2 = logits.clone()
    logits2[:, 0] += 100 * torch.randn(V)
    assert torch.allclose(base, next_token_loss(logits2, x, pad_id=0))
    # Padding à direita também não entra na perda.
    xp = torch.cat([x, torch.zeros(2, 5, dtype=torch.long)], dim=1)
    lp = torch.cat([logits, torch.randn(2, 5, V)], dim=1)
    assert torch.allclose(base, next_token_loss(lp, xp, pad_id=0))


def test_build_model_and_weight_tying():
    m = build_model({"type": "transformer", "vocab_size": V, "pad_id": 0, "n_layers": 1,
                     "d_model": 32, "n_heads": 4, "max_len": 32})
    assert m.head.weight is m.tok_emb.weight
    assert count_params(m) > 0
    with pytest.raises(ValueError):
        build_model({"type": "transformer", "vocab_size": V, "pad_id": 0, "layers": 3})


def test_top_p_and_top_k():
    probs = torch.tensor([[0.05, 0.5, 0.15, 0.3]])
    logits = probs.log()
    kept = torch.isfinite(top_p_filter(logits, 0.75))
    assert kept.tolist() == [[False, True, False, True]]
    assert torch.isfinite(top_p_filter(logits, 0.1)).sum() == 1  # sempre mantém o top-1
    assert torch.isfinite(top_k_filter(logits, 3)).tolist() == [[False, True, True, True]]


def test_cfg_combine():
    cond, unc = torch.randn(3, V), torch.randn(3, V)
    assert torch.allclose(cfg_combine(cond, unc, 1.0), cond)
    assert torch.allclose(cfg_combine(cond, unc, 0.0), unc)
    assert torch.allclose(cfg_combine(cond, unc, 3.0), unc + 3 * (cond - unc))


@pytest.mark.parametrize("make", [tiny_transformer, tiny_lstm])
@pytest.mark.parametrize("gamma", [1.0, 3.0])
def test_generate(make, gamma):
    torch.manual_seed(0)
    model = make(max_len=16) if make is tiny_transformer else make()
    emotions = [4, 5, 6, 7]
    out = generate(model, emotions, SPECIAL, max_new_tokens=40, gamma=gamma, top_p=0.9, seed=1)
    assert len(out) == 4
    banned = set(SPECIAL.banned_during_generation())
    for seq, emo in zip(out, emotions):
        assert seq[:2] == [1, emo]
        assert 2 < len(seq) <= 42
        assert not banned & set(seq[2:])
        assert 2 not in seq[:-1]  # EOS só no fim
    # Mesma semente -> mesmas amostras.
    assert out == generate(model, emotions, SPECIAL, max_new_tokens=40, gamma=gamma,
                           top_p=0.9, seed=1)
    assert model.training  # generate restaura o modo


@pytest.mark.parametrize("make", [tiny_transformer, tiny_lstm])
def test_generate_stops_at_max_bars(make):
    torch.manual_seed(0)
    model = make(max_len=64) if make is tiny_transformer else make()
    bar = 8  # um token qualquer fazendo papel de Bar
    with torch.no_grad():  # força o modelo a gerar Bar com frequência
        model.head.weight.data[bar] += 1.0
    out = generate(model, [4, 5], SPECIAL, max_new_tokens=200, top_p=1.0, seed=0,
                   max_bars=3, bar_id=bar)
    for seq in out:
        assert seq.count(bar) <= 3
        if seq.count(bar) == 3 and len(seq) < 202:
            assert seq[-1] == SPECIAL.eos


@pytest.mark.parametrize("make", [tiny_transformer, tiny_lstm])
def test_overfit_ten_sequences(make):
    """Teste do plano (Etapa 5): a perda deve ir perto de zero em 10 sequências."""
    torch.manual_seed(0)
    model = make()
    x = random_batch(B=10, T=48)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(600):
        loss = next_token_loss(model(x), x, pad_id=0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, loss.item()
