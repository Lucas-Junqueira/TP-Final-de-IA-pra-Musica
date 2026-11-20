"""Testes de src/classifier.py."""

import random

import pytest
import torch

from src.classifier import (
    ClassifierConfig,
    EmotionClassifier,
    _all_windows,
    combine_valence_arousal,
    emotion_report,
    evaluate_sequences,
    load_classifier,
    predict_proba,
    save_classifier,
    train_classifier,
)
from src.sample import SpecialIds

V = 60
SPECIAL = SpecialIds(bos=1, eos=2, pad=0, null=3, emotions=[4, 5, 6, 7])


def tiny(n_classes=4, max_len=32):
    return EmotionClassifier(ClassifierConfig(vocab_size=V, pad_id=0, n_classes=n_classes,
                                              n_layers=2, d_model=32, n_heads=4,
                                              max_len=max_len, dropout=0.0))


def easy_data(n_per=12, seed=0):
    """O quadrante determina a faixa de tokens: fácil de separar."""
    rng = random.Random(seed)
    seqs = []
    for qi, emo in enumerate(SPECIAL.emotions):
        lo = 10 + 10 * qi
        for _ in range(n_per):
            n = rng.randrange(20, 80)  # algumas peças passam do contexto
            seqs.append([1, emo, *[rng.randrange(lo, lo + 10) for _ in range(n)], 2])
    rng.shuffle(seqs)
    return seqs


def test_padding_does_not_change_logits():
    torch.manual_seed(0)
    model = tiny().eval()
    x = torch.randint(8, V, (1, 10))
    padded = torch.cat([x, torch.zeros(1, 6, dtype=torch.long)], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(x), model(padded), atol=1e-5)


def test_emotion_token_is_not_seen():
    torch.manual_seed(0)
    model = tiny()
    a = [1, 4, *range(10, 40), 2]
    b = [1, 6, *range(10, 40), 2]
    assert torch.allclose(*predict_proba(model, [a, b], SPECIAL))


def test_windows_cover_all_events():
    events = list(range(100))
    windows = _all_windows(events, bos=1, max_len=32)
    assert all(w[0] == 1 and len(w) <= 32 for w in windows)
    assert set(t for w in windows for t in w[1:]) == set(events)


def test_emotion_report():
    # Q1->Q1, Q2->Q3 (erra quadrante e arousal), Q3->Q3, Q4->Q1 (erra arousal)
    r = emotion_report([0, 1, 2, 3], [0, 2, 2, 0])
    assert r["acc_4"] == 0.5
    assert r["acc_valence"] == 1.0
    assert r["acc_arousal"] == 0.5
    assert r["confusion"][1][2] == 1 and r["confusion"][3][0] == 1


def test_combine_valence_arousal():
    assert combine_valence_arousal([1, 0, 0, 1], [1, 1, 0, 0]) == [0, 1, 2, 3]


@pytest.mark.parametrize("target,n_classes", [("quadrant", 4), ("valence", 2)])
def test_learns_easy_task(target, n_classes):
    torch.manual_seed(0)
    model = tiny(n_classes=n_classes)
    train, val = easy_data(seed=0), easy_data(n_per=4, seed=1)
    hist = train_classifier(model, train, val, SPECIAL, target=target, epochs=30,
                            batch_size=8, lr=3e-3, patience=10, log=None)
    assert hist["best_val_acc"] == 1.0
    if target == "quadrant":
        assert evaluate_sequences(model, val, SPECIAL)["acc_4"] == 1.0


def test_wrong_number_of_classes():
    with pytest.raises(ValueError):
        train_classifier(tiny(n_classes=4), easy_data(), easy_data(), SPECIAL,
                         target="arousal", log=None)


def test_save_load(tmp_path):
    model = tiny()
    save_classifier(model, tmp_path / "clf.pt", target="quadrant")
    loaded, target = load_classifier(tmp_path / "clf.pt")
    seqs = easy_data(n_per=2)
    assert target == "quadrant"
    assert torch.allclose(predict_proba(model, seqs, SPECIAL), predict_proba(loaded, seqs, SPECIAL))
