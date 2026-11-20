"""Testes de src/markov.py."""

import random

from src.markov import QuadrantMarkov, filter_invalid_remi
from src.sample import SpecialIds

SPECIAL = SpecialIds(bos=1, eos=2, pad=0, null=3, emotions=[4, 5, 6, 7])
BOS, EOS = SPECIAL.bos, SPECIAL.eos


def quadrant_data(n_per=5, length=30, seed=0):
    """Cada quadrante usa um conjunto disjunto de tokens: 10-19, 20-29, 30-39, 40-49."""
    rng = random.Random(seed)
    seqs = []
    for qi, emo in enumerate(SPECIAL.emotions):
        lo = 10 * (qi + 1)
        for _ in range(n_per):
            seqs.append([BOS, emo, *[rng.randrange(lo, lo + 10) for _ in range(length)], EOS])
    seqs.append([BOS, SPECIAL.null, 10, 11, EOS])  # <NULL>: deve ser ignorada
    return seqs


def test_one_chain_per_quadrant():
    model = QuadrantMarkov(order=3, special=SPECIAL).fit(quadrant_data())
    for qi, emo in enumerate(SPECIAL.emotions):
        out = model.generate(emo, max_new_tokens=200, seed=qi)
        assert out[:2] == [BOS, emo]
        lo = 10 * (qi + 1)
        assert all(lo <= t < lo + 10 for t in out[2:] if t != EOS)
        assert EOS not in out[2:-1]


def test_same_seed_same_output():
    model = QuadrantMarkov(order=4, special=SPECIAL).fit(quadrant_data())
    kwargs = dict(max_new_tokens=100, temperature=1.2, top_p=0.9, seed=7)
    assert model.generate(4, **kwargs) == model.generate(4, **kwargs)


def test_high_order_copies_unique_training_sequence():
    seq = [BOS, 4, *range(10, 40), EOS]  # todos os contextos são únicos
    others = [[BOS, e, 50, 51, EOS] for e in SPECIAL.emotions[1:]]
    model = QuadrantMarkov(order=2, special=SPECIAL).fit([seq, *others])
    assert model.generate(4, seed=0) == seq


def test_fallback_to_lower_order():
    seqs = [[BOS, e, 10, 11, 12, EOS] for e in SPECIAL.emotions]
    model = QuadrantMarkov(order=4, special=SPECIAL).fit(seqs)
    # Contexto nunca visto na ordem 4..1 (token 30): cai para a ordem 0.
    counts = model.next_counts(4, [BOS, 30])
    assert counts == model.counts[4][()]
    # Contexto visto só na ordem 1 (12 depois de 30): usa a ordem 1.
    assert model.next_counts(4, [BOS, 30, 12]) == {EOS: 1}


def test_top_p_restricts_to_most_frequent():
    # Depois de <BOS>, o token 10 aparece 9 vezes e o 11 uma vez.
    seqs = [[BOS, 4, 10, EOS]] * 9 + [[BOS, 4, 11, EOS]]
    seqs += [[BOS, e, 10, EOS] for e in SPECIAL.emotions[1:]]
    model = QuadrantMarkov(order=1, special=SPECIAL).fit(seqs)
    firsts = {model.generate(4, seed=s, top_p=0.5)[2] for s in range(30)}
    assert firsts == {10}


def test_max_bars():
    seqs = [[BOS, e, *([10, 11, 12] * 20), EOS] for e in SPECIAL.emotions]
    model = QuadrantMarkov(order=2, special=SPECIAL).fit(seqs)
    out = model.generate(4, max_new_tokens=500, seed=0, max_bars=5, bar_id=10)
    assert out.count(10) == 5 and out[-1] == EOS


def test_save_load(tmp_path):
    model = QuadrantMarkov(order=3, special=SPECIAL).fit(quadrant_data())
    model.save(tmp_path / "m.pkl")
    loaded = QuadrantMarkov.load(tmp_path / "m.pkl")
    assert loaded.generate(5, seed=3) == model.generate(5, seed=3)


def test_filter_keeps_valid_sequence():
    valid = ["Bar_None", "Position_0", "Tempo_120.0", "Pitch_60", "Velocity_63", "Duration_1.0.8",
             "Pitch_64", "Velocity_63", "Duration_0.4.8", "Position_8", "Pitch_67",
             "Velocity_71", "Duration_2.0.8", "Bar_None", "Position_0", "Pitch_60",
             "Velocity_63", "Duration_1.0.8"]
    assert filter_invalid_remi(valid) == (valid, 0)


def test_filter_drops_out_of_order_tokens():
    tokens = [
        "Position_0",                                    # antes do primeiro Bar
        "Bar_None", "Pitch_60",                          # Pitch sem Position
        "Position_4", "Velocity_63",                     # Velocity sem Pitch
        "Pitch_62", "Velocity_63", "Position_6",         # trinca incompleta (2 tokens)
        "Pitch_64", "Velocity_63", "Duration_1.0.8",     # ok
        "Position_2",                                    # Position voltando
        "Duration_1.0.8",                                # Duration solto
        "Bar_None", "Position_0", "Pitch_65", "Velocity_63", "Duration_1.0.8",
        "<EOS>",                                         # token especial
    ]
    kept, dropped = filter_invalid_remi(tokens)
    assert kept == ["Bar_None", "Position_4", "Position_6", "Pitch_64", "Velocity_63",
                    "Duration_1.0.8", "Bar_None", "Position_0", "Pitch_65", "Velocity_63",
                    "Duration_1.0.8"]
    assert dropped == len(tokens) - len(kept)
