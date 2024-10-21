import pytest

from seqinfer.infer.generators import (
    MotifInserter,
    PWMInserter,
    RandomSequenceGenerator,
)


def test_random_sequence_generator():
    rsg = RandomSequenceGenerator("ATCG", seed=42)
    seq = rsg.generate_sequence(10)
    assert len(seq) == 10
    assert set(seq).issubset(set("ATCG"))

    # Test with custom character distribution
    rsg = RandomSequenceGenerator("ATCG", char_distribution=[0.1, 0.2, 0.3, 0.4], seed=42)
    seq = rsg.generate_sequence(1000)
    counts = {char: seq.count(char) for char in "ATCG"}
    assert counts["C"] > counts["A"]  # Expected based on distribution


def test_motif_inserter():
    inserter = MotifInserter("AT-G")
    seq = "GGGGGGGGGG"
    modified_seq = inserter.insert_motif(seq, 3)
    assert modified_seq == "GGGATGGGGG"

    # Test insertion with negative index
    modified_seq = inserter.insert_motif(seq, -4)
    assert modified_seq == "GGGGGGATGG"

    # Test invalid insertion
    with pytest.raises(ValueError):
        inserter.insert_motif(seq, 8)


def test_pwm_inserter():
    pwm = [
        {"A": 0.5, "T": 0.5},
        {"C": 1.0},
        {"G": 0.7, "T": 0.3},
    ]
    pwm_inserter = PWMInserter(pwm, seed=42)
    motif = pwm_inserter.sample_motif()
    assert len(motif) == len(pwm)
    assert set(motif).issubset(set("ACGT"))

    seq = "GGGGGGGGGG"
    modified_seq = pwm_inserter.insert_pwm(seq, 4)
    assert len(modified_seq) == len(seq)

    # Test invalid insertion
    with pytest.raises(ValueError):
        pwm_inserter.insert_pwm(seq, 9)


if __name__ == "__main__":
    pytest.main()
