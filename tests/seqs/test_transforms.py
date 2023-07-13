import re

import numpy as np
import pytest

from seqlearn.seqs.datasets import SeqFromMemDataset
from seqlearn.seqs.transforms import GenerateKmerFreq, GenerateMultiKmerFreq


def test_generate_kmer_freq():
    kmerfreq_generator = GenerateKmerFreq(k=1)
    output = kmerfreq_generator("abca")
    assert all(kmerfreq_generator.kmer_names == np.array(["a", "b", "c"]))
    assert np.all(output == np.array([[2, 1, 1]]))


def test_generate_kmer_freq_for_dataset():
    kmerfreq_generator = GenerateKmerFreq(3, vocabulary=["ABC", "CDE", "LMN"])
    seq_dataset = SeqFromMemDataset(
        sequences=["ABCDEABC", "HIJKLMN"],
        targets=[0, 1],
        transform_sequences=kmerfreq_generator,
    )
    assert all(kmerfreq_generator.kmer_names == np.array(["ABC", "CDE", "LMN"]))
    assert np.all(seq_dataset[0][0] == np.array([[2, 1, 0]]))
    assert np.all(seq_dataset[1][0] == np.array([0, 0, 1]))


def test_generate_multi_kmer_freq():
    multi_kmerfreq_generator = GenerateMultiKmerFreq(
        ks=[1, 2, 4], vocabulary=["a", "b", "c", "d", "bc", "abca"]
    )
    output = multi_kmerfreq_generator("abca")
    assert all(multi_kmerfreq_generator.kmer_names == np.array(["a", "b", "c", "d", "bc", "abca"]))
    assert np.all(output == np.array([[2, 1, 1, 0, 1, 1]]))
    assert multi_kmerfreq_generator.kmer_index_mapping == {
        kmer: i for i, kmer in enumerate(["a", "b", "c", "d", "bc", "abca"])
    }


def test_generate_multi_kmer_freq_no_vocabulary():
    multi_kmerfreq_generator = GenerateMultiKmerFreq(ks=[1, 3])
    output = multi_kmerfreq_generator("abca")
    assert all(multi_kmerfreq_generator.kmer_names == np.array(["a", "b", "c", "abc", "bca"]))
    assert np.all(output == np.array([[2, 1, 1, 1, 1]]))


def test_generate_multi_kmer_raise():
    with pytest.raises(ValueError, match="Input vocabulary doesn't contain any 3mer."):
        GenerateMultiKmerFreq(ks=[1, 3], vocabulary=["a", "b", "c", "d"])

    with pytest.raises(
        ValueError,
        match=re.escape("Input vocabulary contains 2mer outside input range [1, 3]."),
    ):
        GenerateMultiKmerFreq(ks=[1, 3], vocabulary=["a", "bc"])
