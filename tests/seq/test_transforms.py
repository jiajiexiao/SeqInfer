import re

import numpy as np
import pytest
import torch

from seqinfer.seq.datasets import SeqFromMemDataset
from seqinfer.seq.transforms import (
    Compose,
    KmerFreqGenerator,
    KmerTokenizer,
    LabelEncoder,
    MultiKmerFreqGenerator,
    OneHotEncoder,
    ToTensor,
)
from seqinfer.seq.vocabularies import SpecialToken


class TestOneHotEncoder:
    """Unit test class for OneHotEncoder"""

    @pytest.fixture
    def input_ids(self):
        """Pytest fixture for usable input_ids for tests"""
        return [0, 2, 1]

    def test_ohe_without_neutral_tokens(self, input_ids):
        """Test OneHotEncoder without neutral tokens"""
        ohe = OneHotEncoder(vocab_size=4)
        ohe_output = ohe(input_ids)
        expected_output = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
        assert np.all(ohe_output == expected_output)

    def test_ohe_with_neutral_tokens(self, input_ids):
        """Test OneHotEncoder with neutral tokens"""
        ohe = OneHotEncoder(vocab_size=4, neutral_ids=[2])
        ohe_output = ohe(input_ids)
        expected_output = np.array([[1, 0, 0], [1.0 / 3, 1.0 / 3, 1.0 / 3], [0, 1, 0]])
        assert np.allclose(ohe_output, expected_output)

    def test_ohe_invalid_length(self):
        """Test OneHotEncoder with invalid neutral_ids length"""
        with pytest.raises(
            ValueError,
            match=re.escape("len(neutral_ids) should be smaller than vocab_size."),
        ):
            OneHotEncoder(vocab_size=4, neutral_ids=[0, 1, 2, 3])


class TestKmerTokenizer:
    """Unit test class for KmerTokenizer"""

    @pytest.fixture
    def vocab_dict(self):
        """Pytest fixture for usable input arg (i.e. vocab_dict) for tests"""
        return {"A": 0, "C": 1, "G": 2, "T": 3}

    def test_tokenize_single_char_token(self, vocab_dict):
        """Test tokenization for 1mer"""
        tokenizer = KmerTokenizer(k=1, stride=1, vocab_dict=vocab_dict, special_tokens=None)
        seq = "AATTCG"
        expected_tokens = ["A", "A", "T", "T", "C", "G"]
        tokens = tokenizer.tokenize(seq)
        assert tokens == expected_tokens

    def test_tokenize_multiple_char_token(self, vocab_dict):
        """Test tokenization for kmers, k>1"""
        tokenizer = KmerTokenizer(k=3, stride=2, vocab_dict=vocab_dict, special_tokens=None)
        seq = "AATTC"
        expected_tokens = ["AAT", "TTC"]
        tokens = tokenizer.tokenize(seq)
        assert tokens == expected_tokens

    def test_tokenize_truncation(self, vocab_dict, caplog):
        """Test input seq needs to be truncated during tokenization"""
        tokenizer = KmerTokenizer(
            k=1,
            stride=1,
            vocab_dict=vocab_dict,
            num_output_tokens=3,
            special_tokens=None,
        )
        seq = "AATTCG"
        expected_tokens = ["A", "A", "T"]
        tokens = tokenizer.tokenize(seq)
        expected_warning_msg = (
            "The current choices of k=1, stride=1, num_output_tokens=3, "
            "special_tokens=None will leave out the tailed TCG during tokenization. "
            "If this is not desired, consider changing the values of these arguments."
        )

        assert tokens == expected_tokens
        assert expected_warning_msg in caplog.text

    def test_tokenize_padding(self, vocab_dict, caplog):
        """Test padding is needed during tokenization"""
        tokenizer = KmerTokenizer(
            k=1,
            stride=1,
            vocab_dict=vocab_dict,
            num_output_tokens=8,
            special_tokens=SpecialToken,
        )
        seq = "AAT"
        expected_tokens = [
            SpecialToken.CLS,
            "A",
            "A",
            "T",
            SpecialToken.PAD,
            SpecialToken.PAD,
            SpecialToken.PAD,
            SpecialToken.EOS,
        ]
        tokens = tokenizer.tokenize(seq)
        expected_info_msg = "Added CLS, EOS and 3 PAD to tokens."
        assert tokens == expected_tokens
        assert expected_info_msg in caplog.text

    def test_tokenize_no_enough_std_tokens(self, vocab_dict):
        """Test no less standard tokens than desired output length"""
        tokenizer = KmerTokenizer(
            k=1,
            stride=1,
            vocab_dict=vocab_dict,
            num_output_tokens=8,
            special_tokens=None,
        )
        with pytest.raises(
            ValueError,
            match=(
                "The number of output tokens 3 is smaller than the desired output length 8. "
                "Padding from special_tokens is needed in this case."
            ),
        ):
            tokenizer.tokenize("AAT")

    def test_conversion_token_to_idx(self, vocab_dict):
        """Test converting token to idx"""
        tokenizer = KmerTokenizer(k=1, stride=1, vocab_dict=vocab_dict)
        tokens = ["A", "C", "G", "T"]
        expected_ids = [0, 1, 2, 3]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        assert ids == expected_ids

    def test_conversion_id_to_token(self, vocab_dict):
        """Test converting idx to token"""
        tokenizer = KmerTokenizer(k=1, stride=1, vocab_dict=vocab_dict)
        ids = [4, 0, 1, 2, 3, 6, 7, 8, 5]
        expected_tokens = ["<cls>", "A", "C", "G", "T", "<unk>", "<pad>", "<mask>", "<eos>"]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        assert tokens == expected_tokens

    def test_token_to_idx_unknown_token(self, vocab_dict):
        """Test converting unknown token to unknown token idx"""
        tokenizer = KmerTokenizer(k=1, stride=1, vocab_dict=vocab_dict)
        token = "N"
        expected_id = 6  # unk_token ID
        token_id = tokenizer.token_to_idx(token)
        assert token_id == expected_id

        tokenizer = KmerTokenizer(k=1, stride=1, vocab_dict=vocab_dict, special_tokens=None)
        with pytest.raises(
            ValueError,
            match=(
                "N is not found in specified vocabulary. If this is allowed, special_tokens "
                "with unknown token is needed."
            ),
        ):
            tokenizer.token_to_idx(token)

    def test_idx_to_token_unknown_idx(self, vocab_dict):
        """Test converting unknown token idx for unknown token"""
        tokenizer = KmerTokenizer(k=1, stride=1, vocab_dict=vocab_dict)
        token_id = 6  # unk_token ID
        expected_token = "<unk>"
        token = tokenizer.idx_to_token(token_id)
        assert token == expected_token

    def test_call_method(self, vocab_dict):
        """Test calling KmerTokenizer"""
        tokenizer = KmerTokenizer(k=1, stride=2, vocab_dict=vocab_dict)
        tokenizer_no_special_tokens = KmerTokenizer(
            k=1, stride=2, vocab_dict=vocab_dict, special_tokens=None
        )
        seq = "ACTG"
        assert tokenizer(seq) == [4, 0, 3, 5]
        # k=1, stride=2, thus C, G were ignored. "ACTG"->["<cls>", "A", "T", "<eos>"]->[4, 0, 3, 5]
        assert tokenizer_no_special_tokens(seq) == [0, 3]


class TestKmerFreqGenerator:
    """Unit test class for KmerFreqGenerator"""

    def test_kmer_freq_generator_direct(self):
        """Test the use of KmerFreqGenerator directly"""
        kmerfreq_generator = KmerFreqGenerator(k=1)
        output = kmerfreq_generator("abca")
        assert all(kmerfreq_generator.kmer_names == np.array(["a", "b", "c"]))
        assert np.all(output == np.array([[2, 1, 1]]))

    def test_kmer_freq_generator_for_dataset(self):
        """Test KmerFreqGenerator for its use in dataset"""
        kmerfreq_generator = KmerFreqGenerator(3, vocabulary=["ABC", "CDE", "LMN"])
        seq_dataset = SeqFromMemDataset(
            sequences=["ABCDEABC", "HIJKLMN"],
            targets=[0, 1],
            transform_sequences=kmerfreq_generator,
        )
        assert all(kmerfreq_generator.kmer_names == np.array(["ABC", "CDE", "LMN"]))
        assert np.all(seq_dataset[0][0] == np.array([[2, 1, 0]]))
        assert np.all(seq_dataset[1][0] == np.array([0, 0, 1]))


class TestMultiKmerFreqGenerator:
    """Unit test class for MultiKmerFreqGenerator"""

    def test_multi_kmer_freq_generator_with_vocabulary(self):
        """Test MultiKmerFreqGenerator with vocabulary"""
        multi_kmerfreq_generator = MultiKmerFreqGenerator(
            ks=[1, 2, 4], vocabulary=["a", "b", "c", "d", "bc", "abca"]
        )
        output = multi_kmerfreq_generator("abca")
        assert all(
            multi_kmerfreq_generator.kmer_names == np.array(["a", "b", "c", "d", "bc", "abca"])
        )
        assert np.all(output == np.array([[2, 1, 1, 0, 1, 1]]))
        assert multi_kmerfreq_generator.kmer_index_mapping == {
            kmer: i for i, kmer in enumerate(["a", "b", "c", "d", "bc", "abca"])
        }

    def test_multi_kmer_freq_generator_without_vocabulary(self):
        """Test MultiKmerFreqGenerator without vocabulary"""
        multi_kmerfreq_generator = MultiKmerFreqGenerator(ks=[1, 3])
        output = multi_kmerfreq_generator("abca")
        assert all(multi_kmerfreq_generator.kmer_names == np.array(["a", "b", "c", "abc", "bca"]))
        assert np.all(output == np.array([[2, 1, 1, 1, 1]]))

    def test_multi_kmer_freq_generator_raise(self):
        """Test raise errors for MultiKmerFreqGenerator"""
        with pytest.raises(ValueError, match="Input vocabulary doesn't contain any 3mer."):
            MultiKmerFreqGenerator(ks=[1, 3], vocabulary=["a", "b", "c", "d"])

        with pytest.raises(
            ValueError,
            match=re.escape("Input vocabulary contains 2mer outside input range [1, 3]."),
        ):
            MultiKmerFreqGenerator(ks=[1, 3], vocabulary=["a", "bc"])


class TestToTensor:
    """Unit test class for ToTensor"""

    def test_to_tensor_with_desired_dtype(self):
        """Test ToTensor with desired dtype"""
        output = ToTensor(torch.float16)([1, 2, 3])
        expected_output = torch.tensor([1, 2, 3], dtype=torch.float16)
        assert torch.equal(output, expected_output)
        assert output.dtype == expected_output.dtype

    def test_to_tensor_inferred_dtype(self):
        """Test ToTensor without desired dtype"""
        output = ToTensor()([1, 2, 3])
        expected_output = torch.tensor([1, 2, 3], dtype=torch.int64)
        assert torch.equal(output, expected_output)
        assert output.dtype == expected_output.dtype


class TestCompose:
    """Unit tests for the Compose class."""

    def transform_fn1(self, data: float) -> float:
        """Sample transform function that adds 1 to the input data."""
        return data + 1

    def transform_fn2(self, data: float) -> float:
        """Sample transform function that multiplies the input data by 2."""
        return data * 2

    def test_compose_with_single_transform(self):
        """Test applying a single transform using the Compose class."""
        input_data = 10
        transforms = [self.transform_fn1]
        composed_transforms = Compose(transforms)
        output = composed_transforms(input_data)
        assert output == self.transform_fn1(input_data)
        assert repr(composed_transforms) == "Compose(transform_fn1)"

    def test_compose_with_multiple_transforms(self):
        """Test applying multiple transforms using the Compose class."""
        input_data = 10
        transforms = [self.transform_fn1, self.transform_fn2]
        composed_transforms = Compose(transforms)

        output = composed_transforms(input_data)
        expected_output = self.transform_fn2(self.transform_fn1(input_data))
        assert output == expected_output
        assert repr(composed_transforms) == "Compose(transform_fn1, transform_fn2)"

    def test_compose_with_noncallable_transforms(self):
        """Test that Compose correctly raises an assertion error for non-callable transforms."""
        transforms = [self.transform_fn1, "non_callable"]
        with pytest.raises(
            AssertionError,
            match=f"The {1}th transform non_callable is not callable.",
        ):
            Compose(transforms)


class TestLabelEncoder:
    """Unit test class for LabelEncoder"""

    @pytest.fixture
    def le(self):
        """Fixture to create LabelEncoder instance"""
        mapping = {"cat": 0, "dog": 1, "bird": 2}
        return LabelEncoder(mapping)

    def test_init(self, le):
        """Test initialization stores mapping correctly"""
        assert le.mapping == {"cat": 0, "dog": 1, "bird": 2}

    def test_call(self, le):
        """Test encoding known labels to integers"""
        labels = ["cat", "dog", "bird"]
        expected = [0, 1, 2]
        assert all(le(labels) == expected)

    def test_missing_label(self, le):
        """Test error raised when unknown label is passed"""
        labels = ["cat", "dog", "fish"]
        with pytest.raises(KeyError):
            le(labels)

    def test_empty(self, le):
        """Test encoding empty label list"""
        labels = []
        expected = np.array([])
        assert all(le(labels) == expected)
