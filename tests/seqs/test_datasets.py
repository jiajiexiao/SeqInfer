# import torch

# from torchvision.transforms import Compose, ToTensor
from seqlearn.seqs.datasets import SeqFromFileDataset, SeqFromMemDataset


def test_seq_from_memory_dataset():
    seq_dataset = SeqFromMemDataset(
        sequences=["ABCDEFG", "HIJKLMN"],
        targets=[0, 1],
    )
    assert len(seq_dataset) == 2
    assert seq_dataset[1] == ("HIJKLMN", 1)


def test_seq_from_file_dataset():
    seq_dataset = SeqFromFileDataset(
        seq_file="examples/aptamers/pos.fasta",
        seq_file_fmt="fasta",
    )
    assert len(seq_dataset) == 2500
    assert seq_dataset[9], None == ("ATATTGAACTCCT", None)
