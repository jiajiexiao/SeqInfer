"""
Define dataset class
Transform functions such as kmer encoding, one-hot encoding, etc 
"""
# pwm
from typing import Any, Callable

from Bio import SeqIO
from torch.utils.data import Dataset


class SeqFromMemDataset(Dataset):
    """Seq dataset from memory"""

    def __init__(
        self,
        sequences: list[str],
        targets: None | list[Any] = None,
        transform_sequences: None | Callable = None,
        transform_targets: None | Callable = None,
    ) -> None:
        """Constructor method for SeqFromMemDataset.

        Args:
            sequences (list[str]):
                Input sequences
            targets (None | list[Any], optional):
                Input targets correspond to each sequence. Defaults to None.
            transform_sequences (None | Callable, optional):
                transformation on each sequence record. Defaults to None.
            transform_targets (None | Callable, optional):
                transformation on each target record. Defaults to None.
        """

        super().__init__()
        self.sequences = sequences
        self.targets = targets
        if targets:
            assert len(self.sequences) == len(self.targets)
        self.transform_sequences = transform_sequences
        self.transform_targets = transform_targets

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[str, Any]:
        seq = self.sequences[index]
        if self.transform_sequences:
            seq = self.transform_sequences(seq)

        target = self.targets[index] if self.targets else self.targets
        if self.transform_targets:
            target = self.transform_targets(target)
        return seq, target


class SeqFromFileDataset(Dataset):
    """Seq dataset from a seq file"""

    def __init__(
        self,
        seq_file: str,
        seq_file_fmt: str,
        targets: None | list[Any] = None,
        transform_sequences: None | Callable = None,
        transform_targets: None | Callable = None,
    ) -> None:
        """Constructor method for SeqFromFileDataset

        Args:
            seq_file (str):
                path for the input seq file.
            seq_file_fmt (str):
                format of the input seq file. Should be supported by Biopython's Bio.SeqIO.
            targets (None | list[Any], optional):
                Input targets correspond to each sequence. Defaults to None.
            transform_sequences (None | Callable, optional):
                transformation on each sequence record. Defaults to None.
            transform_targets (None | Callable, optional):
                transformation on each target record. Defaults to None.
        """
        self.seq_file = seq_file
        self.seq_file_fmt = seq_file_fmt
        self.targets = targets

        self.transform_sequences = transform_sequences
        self.transform_targets = transform_targets

    def __len__(self) -> int:
        # not loading everything from the seq_file to avoid potential OOM errors.
        return len(SeqIO.index(self.seq_file, self.seq_file_fmt))

    def __getitem__(self, index: int) -> tuple[str, Any]:
        for i, record in enumerate(SeqIO.parse(self.seq_file, self.seq_file_fmt)):
            if i == index:
                seq = str(record.seq)
                break
        if self.transform_sequences:
            seq = self.transform_sequences(seq)

        target = self.targets[index] if self.targets else self.targets
        if self.transform_targets:
            target = self.transform_targets(target)
        return seq, target
