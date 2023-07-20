import numpy as np
import torch
from torchvision.transforms import Compose

from seqlearn.seqs.datasets import SeqFromFileDataset, SeqFromMemDataset
from seqlearn.seqs.transforms import KmerTokenizer, OneHotEncoder, ToTensor
from seqlearn.seqs.vocabularies import unambiguous_dna_vocabulary_dict


class TestSeqFromMemDataset:
    def test_seq_from_memory_dataset(self):
        seq_dataset = SeqFromMemDataset(
            sequences=["ABCDEFG", "HIJKLMN"],
            targets=[0, 1],
        )
        assert len(seq_dataset) == 2
        assert seq_dataset[1] == ("HIJKLMN", 1)


class TestSeqFromFileDataset:
    def test_seq_from_file_dataset(self):
        seq_dataset = SeqFromFileDataset(
            seq_file="examples/aptamers/pos.fasta",
            seq_file_fmt="fasta",
        )
        assert len(seq_dataset) == 2500
        assert seq_dataset[9] == ("ATATTGAACTCCT", None)

    def test_transforms_for_seq_from_file_dataset(self):
        seq_dataset = SeqFromFileDataset(
            seq_file="examples/aptamers/pos.fasta",
            seq_file_fmt="fasta",
            transform_sequences=Compose(
                [
                    KmerTokenizer(
                        k=1,
                        stride=1,
                        vocab_dict=unambiguous_dna_vocabulary_dict,
                        num_output_tokens=3,
                        special_tokens=None,
                    ),
                    OneHotEncoder(vocab_size=len(unambiguous_dna_vocabulary_dict)),
                    ToTensor(),
                ]
            ),
        )
        assert len(seq_dataset) == 2500
        assert torch.equal(
            seq_dataset[9][0],
            torch.from_numpy(
                np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            ),
        )
