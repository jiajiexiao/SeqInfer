import random

import numpy as np


class RandomSequenceGenerator:
    def __init__(
        self,
        alphabet: str,
        char_distribution: list[float] | None = None,
        seed: int | None = None,
    ):
        """
        Initialize the random sequence generator with a desired character distribution and optional
        random seed.

        Args:
            alphabet (str): A string representing the possible characters in the sequence (e.g.,
            "ATCG").
            char_distribution (Optional[list[float]]): A list representing the desired probability
            distribution for each character in the alphabet.
            seed (Optional[int]): The seed for random number generation to ensure reproducibility.
        """
        self.alphabet = alphabet
        self.char_distribution = (
            char_distribution if char_distribution else [1 / len(alphabet)] * len(alphabet)
        )
        self.rng = random.Random(seed)

    def generate_sequence(self, length: int) -> str:
        """
        Generate a random sequence of a given length with the specified character distribution.
        Args:
            length (int): The length of the sequence to be generated.
        Returns:
            str: A randomly generated sequence following the desired character distribution.
        """
        return "".join(self.rng.choices(self.alphabet, weights=self.char_distribution, k=length))


class MotifInserter:
    def __init__(self, motif: str, gap_char: str = "-"):
        """
        Initialize the motif inserter.

        Args:
            motif (str): The motif to be inserted into the sequence, where gaps are represented by
            a special character (e.g., '-').

            gap_char (str): The character used to represent gaps in the motif (default is '-').
        """
        self.motif = motif
        self.motif_len = len(motif)
        self.gap_char = gap_char

    def insert_motif(self, seq: str, insert_pos: int) -> str:
        """
        Insert the motif (including gaps) into the given sequence at the specified positions.

        Args:
            seq (str): The sequence into which the motif will be inserted.
            insert_pos (int): The position in the sequence to insert the motif.

        Returns:
            str: The modified sequence with the motif inserted.
        """
        seq_len = len(seq)
        seq = list(seq)

        if insert_pos < 0:
            insert_pos += seq_len

        if insert_pos < 0 or insert_pos + self.motif_len > seq_len:
            raise ValueError(
                f"Insert pos {insert_pos} is not valid for input sequence of length {seq_len}"
            )

        # Insert the motif, excluding gaps
        for motif_idx, seq_idx in enumerate(range(insert_pos, insert_pos + self.motif_len)):
            if self.motif[motif_idx] != self.gap_char:
                seq[seq_idx] = self.motif[motif_idx]

        return "".join(seq)


class PWMInserter:
    def __init__(self, pwm: list[dict[str, float]], gap_char: str = "-", seed: int | None = None):
        """
        Initialize the PWM inserter.

        Args:
            pwm (list[dict[str, float]]): The position weight matrix for probability of possible
            chars in each position.

            gap_char (str): The character used to represent gaps in the motif (default is '-').

            seed (Optional[int]): The seed for random number generation to ensure reproducibility.
        """
        self.pwm = pwm
        self.pwm_len = len(pwm)
        self.gap_char = gap_char
        self.rng = random.Random(seed)

    def sample_motif(self) -> str:
        """
        Sample a motif based on the PWM probabilities for each position.

        Returns:
            str: A sampled motif.
        """
        motif = []
        for position_probs in self.pwm:
            chars, probs = zip(*position_probs.items())
            sampled_char = self.rng.choices(population=chars, weights=probs)
            motif.append(sampled_char)
        return "".join(motif)

    def insert_pwm(self, seq: str, insert_pos: int) -> str:
        """
        Insert a PWM-generated motif into the given sequence at the specified position.

        Args:
            seq (str): The sequence into which the motif will be inserted.
            insert_pos (int): The position in the sequence where the PWM motif will be inserted.

        Returns:
            str: The modified sequence with the PWM motif inserted.
        """
        seq_len = len(seq)
        seq = list(seq)

        if insert_pos < 0:
            insert_pos += seq_len

        if insert_pos < 0 or insert_pos + self.pwm_len > seq_len:
            raise ValueError(
                f"Insert pos {insert_pos} is not valid for input sequence of length {seq_len}"
            )

        # Sample a motif based on PWM probabilities
        motif = self.pwm.sample_motif()

        # Insert the motif into the sequence
        for motif_idx, seq_idx in enumerate(range(insert_pos, insert_pos + self.pwm_len)):
            if motif[motif_idx] != self.gap_char:
                seq[seq_idx] = motif[motif_idx]

        return "".join(seq)
