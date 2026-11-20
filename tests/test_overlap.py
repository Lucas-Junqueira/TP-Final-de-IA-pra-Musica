"""Testes de src/overlap.py."""

import random

from symusic import Note, Score, Tempo, Track

from src.overlap import chord_sequence, fingerprints


def write_song(path, pitches, shift=0):
    s = Score(480)
    s.tempos.append(Tempo(0, 120.0))
    t = Track(program=0)
    for i, p in enumerate(pitches):
        t.notes.append(Note(i * 240, 240, p + shift, 80))
        t.notes.append(Note(i * 240, 240, p + shift - 12, 60))  # acorde: oitava abaixo
    s.tracks.append(t)
    s.dump_midi(str(path))
    return path


def overlap(a, b, kind):
    fa, fb = fingerprints(a), fingerprints(b)
    return len(fa[kind] & fb[kind]) / len(fa[kind])


def test_chords_group_simultaneous_notes(tmp_path):
    chords = chord_sequence(write_song(tmp_path / "a.mid", [60, 62, 64]))
    assert chords == [(48, 60), (50, 62), (52, 64)]


def test_same_song_scores_high_and_random_low(tmp_path):
    rng = random.Random(0)
    melody = [rng.randrange(55, 80) for _ in range(80)]
    other = [rng.randrange(55, 80) for _ in range(80)]
    a = chord_sequence(write_song(tmp_path / "a.mid", melody))
    same = chord_sequence(write_song(tmp_path / "b.mid", melody))
    transposed = chord_sequence(write_song(tmp_path / "c.mid", melody, shift=3))
    unrelated = chord_sequence(write_song(tmp_path / "d.mid", other))
    assert overlap(a, same, "gravacao") == 1.0
    assert overlap(a, transposed, "gravacao") == 0.0
    assert overlap(a, transposed, "melodia") == 1.0  # intervalos não dependem do tom
    assert overlap(a, unrelated, "melodia") < 0.05
