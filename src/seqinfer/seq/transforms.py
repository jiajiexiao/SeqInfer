from typing import Any, Callable, Iterable, Mapping, Type

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

from seqinfer.seq.vocabularies import SpecialToken
from seqinfer.utils.logger import LogLevel, log_msg


class OneHotEncoder:
    """OneHotEncoder class for converting token IDs into one-hot encoded vectors.

    This class is useful for converting categorical token IDs into their corresponding one-hot
    encoded representations. Moreover, it also supports special conversion of `neutral tokens`,
    which will be assigned equal probabilities across all tokens in the vocabulary. This ensures
    that neutral tokens do not dominate the learned representations when used in various downstream
    tasks.
    """

    def __init__(self, vocab_size: int, neutral_ids: list[int] | None = None) -> None:
        """Initialize the OneHotEncoder.

        Args:
            vocab_size (int):
                The size of the vocabulary. It represents the total number of unique tokens in the
                vocabulary. It should also count for the special tokens if they are used during
                tokenization.
            neutral_ids (list[int] | None, optional):
                A optional list of integers representing token IDs that are considered `neutral`.
                For neutral tokens, the one-hot encoding will assign equal probabilities across all
                tokens in the vocabulary. If not provided (None), regular one-hot encoding will be
                used for all token IDs. Defaults to None.

        Raises:
            ValueError: If len(neutral_ids) is greater than or equal to vocab_size.
        """
        assert vocab_size > 0, f"vocab_size needs to be a positive integer but got {vocab_size}."
        self.neutral_ids = neutral_ids

        if self.neutral_ids:
            if len(self.neutral_ids) >= vocab_size:
                raise ValueError("len(neutral_ids) should be smaller than vocab_size.")

            self.vocab_size = vocab_size - len(self.neutral_ids)
        else:
            self.vocab_size = vocab_size

    def __call__(self, ids: list[int]) -> np.ndarray:
        """Encode a list of token IDs into one-hot encoded vectors.

        Args:
            ids (list[int]): A list of integers representing token IDs that need to be one-hot
            encoded.

        Returns:
            np.ndarray: A 2D NumPy array representing the one-hot encoded matrix for the input list
            of token IDs. The shape of the output matrix will be (num_tokens, vocab_size).
        """
        num_tokens = len(ids)
        output = np.zeros((num_tokens, self.vocab_size))
        output[np.arange(num_tokens), ids] = 1.0
        if self.neutral_ids:
            neutral_ids = np.arange(num_tokens)[np.isin(ids, self.neutral_ids)]
            if len(neutral_ids) > 0:
                output[neutral_ids, :] = 1.0 / self.vocab_size
        return output


class KmerTokenizer:
    """Tokenizer based on kmers.

    The KmerTokenizer class provides functionality to tokenize sequences into k-mer tokens. Each
    k-mer token represents a contiguous subsequence of length k in the input sequence. The tokens
    can be overlapping or independent kmers, depending on the specified stride value. The class
    supports conversion between tokens and their corresponding indices using a vocabulary
    dictionary.

    Example use cases with different parameters:

    1. k=1, stride=1, num_output_tokens=None, special_tokens=None:
        - Each token represents a single character in the sequence.
        - No control on output token lengths and no special tokens are added.
        - Example: Tokenizing a DNA sequence "ATCG" results in ['A', 'T', 'C', 'G'].

    2. k=2, stride=2, num_output_tokens=None, special_tokens=None:
        - Each token represents a 2mer in the sequence.
        - The stride value of 2 ensures non-overlapping tokens.
        - No control on output token lengths and no special tokens are added.
        - Example: Tokenizing a DNA sequence "ATCG" results in ['AT', 'CG'].

    3. k=3, stride=1, num_output_tokens=5, special_tokens=SpecialToken:
        - Each token represents a 3-character kmer in the sequence.
        - Total of 5 tokens (including special tokens) are outputted.
        - Special tokens such as CLS (classification), PAD (padding), and EOS (end of sequence) are
          added.
        - Example: Tokenizing a DNA sequence "ATCG" results in ['CLS', 'ATC', 'TCG', 'PAD', 'EOS'].

    Note:
        If `num_output_tokens` is specified and its value is smaller than the total number of kmers
        in the sequence, the tokenization process may ignore the tailed characters. This happens
        when the specified number of output tokens is reached before processing the entire sequence.
        If preservation of all characters is desired, consider adjusting the values of `k`,
        `stride`, and `num_output_tokens` accordingly.
    """

    def __init__(
        self,
        k: int,
        stride: int,
        vocab_dict: dict[str, int],
        num_output_tokens: int | None = None,
        special_tokens: Type[SpecialToken] | None = SpecialToken,
    ) -> None:
        """Initialize the KmerTokenizer.

        Args:
            k (int): Length of each k-mer token. stride (int): Stride value for sliding the k-mer
            window. vocab_dict (dict[str, int]): Vocabulary dictionary mapping tokens to their
            indices. num_output_tokens (int | None, optional):
                Number of tokens (including special tokens) to output. Defaults to None, meaning no
                restrictions on number of tokens to output.
            special_tokens (Type[SpecialToken], optional):
                Special token Enum or None. When special_tokens = None, no special_tokens (e.g. PAD,
                CLS) will be added. Default to the built-in SpecialToken.

        Raises:
            AssertionError: If k is not greater than 0 or stride is not greater than 0.
            AssertionError: If num_output_tokens is not greater than 0 if num_output_tokens != None.
        """

        assert k > 0, f"Each token should contain at least 1 char but input k is {k}"
        assert stride > 0, f"Stride should be at least 1 to slide the kmer window but got {stride}"
        if num_output_tokens is not None:
            assert (
                num_output_tokens > 0
            ), f"num_output_tokens should greater than 1 but got {num_output_tokens}"

        self.k = k
        self.stride = stride
        self.vocab_dict = vocab_dict
        self.num_output_tokens = num_output_tokens
        self._num_std_tokens = (
            self.num_output_tokens
        )  # Num of standard (i.e. non-special) tokens to extract
        self.special_tokens = special_tokens

        if self.special_tokens:
            vocab_size = len(self.vocab_dict)
            # update vocab_dict to include special tokens
            self.vocab_dict.update(
                {
                    self.special_tokens.CLS: vocab_size,
                    self.special_tokens.EOS: vocab_size + 1,
                    self.special_tokens.UNK: vocab_size + 2,
                    self.special_tokens.PAD: vocab_size + 3,
                    self.special_tokens.MASK: vocab_size + 4,
                }
            )

            if self._num_std_tokens is not None:
                # when special_tokens is not None, the cls_token and eos_token will be appended to
                # the head and tail of the token list. Thus the number of non-special tokens needs
                # to be reduced by 2.
                self._num_std_tokens -= 2

        self.idx2token_dict = {idx: token for token, idx in self.vocab_dict.items()}

    def tokenize(self, seq: str) -> list[str]:
        """Tokenize the input sequence into k-mer tokens.

        This method requires no special tokens in the input sequence, and special tokens will be
        added to the output of this method if self.special_tokens != None.

        Args:
            seq (str): The sequence to be tokenized.

        Returns:
            list[str]: List of k-mer tokens.

        """
        tokens = self._generate_std_tokens(seq)
        if self.special_tokens:
            tokens = self._add_special_tokens(tokens)
        else:
            self._check_output_length(tokens)
        return tokens

    def _generate_std_tokens(self, seq: str) -> list[str]:
        """Generate standard k-mer tokens from the input sequence.

        This method requires no special tokens in the input sequence because each char in the seq
        will be treated as the base element in the kmer.

        Args:
            seq (str): The input sequence.

        Returns:
            list[str]: List of standard k-mer tokens.

        """
        tokens = []
        seq_len = len(seq)
        end_idx = 0
        for i in range(int((seq_len - self.k) / self.stride) + 1):
            if (self._num_std_tokens is not None) and (i == self._num_std_tokens):
                break
            start_idx = i * self.stride
            end_idx = start_idx + self.k
            tokens.append(seq[start_idx:end_idx])

        if end_idx < seq_len:
            msg = (
                f"The current choices of k={self.k}, stride={self.stride}, num_output_tokens="
                f"{self.num_output_tokens}, special_tokens={self.special_tokens} "
                f"will leave out the tailed {seq[end_idx: seq_len]} during tokenization. "
                "If this is not desired, consider changing the values of these arguments."
            )
            log_msg(
                name=f"{self.__class__.__name__}.{self.tokenize.__name__}",
                msg=msg,
                log_level=LogLevel.WARNING,
            )
        return tokens

    def _add_special_tokens(self, tokens: list[str]) -> list[str]:
        """Add special tokens to the list of tokens.

        Args:
            tokens (list[str]): The list of tokens to add special tokens to.

        Returns:
            list[str]: List of tokens with special tokens added.

        """
        num_paddings = 0 if self._num_std_tokens is None else self._num_std_tokens - len(tokens)
        tokens = (
            [self.special_tokens.CLS]
            + tokens
            + [self.special_tokens.PAD] * num_paddings
            + [self.special_tokens.EOS]
        )
        log_msg(
            name=f"{self.__class__.__name__}.{self._add_special_tokens.__name__}",
            msg=f"Added CLS, EOS and {num_paddings} PAD to tokens.",
            log_level=LogLevel.INFO,
        )
        return tokens

    def _check_output_length(self, tokens: list[str]) -> None:
        """Check if the number of output tokens matches the desired output length.

        If the desired number of output tokens is specified and the actual number of tokens is
        smaller, a ValueError is raised.

        Args:
            tokens (list[str]): The list of tokens.

        Raises:
            ValueError: If the number of output tokens is smaller than the desired output length.

        """
        if self._num_std_tokens is not None and len(tokens) < self._num_std_tokens:
            raise ValueError(
                f"The number of output tokens {len(tokens)} is smaller than the desired output "
                f"length {self._num_std_tokens}. Padding from special_tokens is needed in this "
                "case."
            )

    def token_to_idx(self, token: str) -> int:
        """Maps a token to its corresponding index in the vocabulary.

        Args:
            token (str): The token to be mapped.

        Returns:
            int: The index of the token in the vocabulary.

        Raises:
            ValueError: If the token is not found in the specified vocabulary and
            special_tokens=None.
        """

        if token in self.vocab_dict:
            idx = self.vocab_dict[token]
        else:
            msg = f"{token} is not found in specified vocabulary."
            log_msg(
                name=f"{self.__class__.__name__}.{self.token_to_idx.__name__}",
                msg=msg,
                log_level=LogLevel.WARNING,
            )

            if not self.special_tokens:
                raise ValueError(
                    msg + " If this is allowed, special_tokens with unknown token is needed."
                )
            idx = self.vocab_dict[self.special_tokens.UNK]
        return idx

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Converts a list of tokens to their corresponding indices in the vocabulary.

        Args:
            tokens (list[str]): The list of tokens to be converted.

        Returns:
            list[int]: The list of indices corresponding to the input tokens.
        """
        return [self.token_to_idx(token) for token in tokens]

    def idx_to_token(self, idx: int) -> str:
        """Maps an index to its corresponding token in the vocabulary.

        Args:
            idx (int): The index to be mapped. offset (int, optional):
                offset integer used in vocabulary dict. Default to 1, to reserve the 0th index for
                CLS token.

        Returns:
            str: The token corresponding to the input index.
        """

        return self.idx2token_dict[idx]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Converts a list of indices to their corresponding tokens in the vocabulary.

        Args:
            ids (list[int]): The list of indices to be converted.

        Returns:
            list[str]: The list of tokens corresponding to the input indices.
        """
        return [self.idx_to_token(idx) for idx in ids]

    def __call__(self, seq: str) -> list[int]:
        """Tokenizes and converts the input sequence into a list of token indices.

        Args:
            seq (str): The sequence to be tokenized and converted.

        Returns:
            list[int]: The list of token indices representing the input sequence.
        """
        outputs = self.tokenize(seq)
        outputs = self.convert_tokens_to_ids(outputs)
        return outputs


class KmerFreqGenerator:
    """Kmer frequency generatior"""

    def __init__(
        self,
        k: int,
        vocabulary: Mapping | Iterable | None = None,
        to_array: bool = True,
    ) -> None:
        """Initialize the KmerFreqGenerator

        Args:
            k (int): length of kmer. vocabulary (Mapping | Iterable, optional):
                vocabulary for kmers of interest. Either a Mapping (e.g., a dict) where keys are
                terms and values are indices in the feature matrix, or an iterable over terms. If
                not given, a vocabulary is determined from the inputs. Indices in the mapping should
                not be repeated and should not have any gap between 0 and the largest index.
                Defaults to None.
            to_array (bool, optional):
                whether to convert sparse matrix to dense one. Defaults to True.
        """
        assert k > 0, f"kmer has to be at least 1-char long but got {k}."
        self.k = k
        self.vocabulary = vocabulary
        self.vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(self.k, self.k),
            vocabulary=self.vocabulary,
            lowercase=False,
        )
        self.to_array = to_array

    def __call__(self, seq: str) -> np.ndarray:
        # no fit happens when vocabulary is not None
        output = self.vectorizer.fit_transform([seq])
        if self.to_array and not isinstance(output, np.ndarray):
            output = output.toarray()
        return output

    @property
    def kmer_names(self) -> np.ndarray:
        """kmer name for corresponding columns"""
        return self.vectorizer.get_feature_names_out()

    @property
    def kmer_index_mapping(self) -> dict[str, int]:
        """A mapping of kmers to feature indices"""
        return self.vectorizer.vocabulary_


class MultiKmerFreqGenerator:
    """Class to generate frequency of kmers with multiple values of k"""

    def __init__(
        self,
        ks: list[int],
        vocabulary: Mapping | Iterable | None = None,
        to_array: bool = True,
    ):
        """Initialize the MultiKmerFreqGenerator

        Args:
            ks (list[int]): values of k to count kmers. vocabulary (Mapping | Iterable | None,
            optional):
                vocabulary for kmers of interest. Check KmerFreqGenerator and sklearn's
                CountVectorizer to see more details about this argument. Defaults to None.
            to_array (bool, optional):
                whether to convert sparse matrix to dense one. Defaults to True.
        """
        self.ks = ks
        self.to_array = to_array
        self.vocab_k = self._get_vocab_k(vocabulary)
        self.kmer_freq_generators = [
            KmerFreqGenerator(k, vocabulary=self.vocab_k[k], to_array=self.to_array)
            for k in self.ks
        ]

    def __call__(self, seq: str) -> np.ndarray:
        output = np.concatenate(
            [generator(seq) for generator in self.kmer_freq_generators], axis=1
        )
        return output

    @property
    def kmer_names(self) -> np.ndarray:
        """kmer name for corresponding columns"""
        return np.concatenate([generator.kmer_names for generator in self.kmer_freq_generators])

    @property
    def kmer_index_mapping(self) -> dict[str, int]:
        """A mapping of kmers to feature indices"""
        return {kmer: i for i, kmer in enumerate(self.kmer_names)}

    def _get_vocab_k(
        self, vocabulary: Mapping | Iterable | None
    ) -> dict[int, Mapping | Iterable | None]:
        """Helper method to parse input vocabulary for each possible value of k

        Args:
            vocabulary (Mapping | Iterable | None):
                vocabulary for kmers of interest. Check KmerFreqGenerator and sklearn's
                CountVectorizer to see more details about this argument.

        Raises:
            ValueError: A kmer in the vocabulary has a length outside the range of ks. ValueError: A
            value of k in ks is not found in the user-specified vocabulary.

        Returns:
            dict[int, Mapping | Iterable | None]: a dictionary contains sub-vocabulary per k.
        """
        if vocabulary:
            vocab_k = {k: [] for k in self.ks}
            for kmer in vocabulary:
                k = len(kmer)
                if k in self.ks:
                    vocab_k[k].append(kmer)
                else:
                    raise ValueError(
                        f"Input vocabulary contains {k}mer outside input range {self.ks}."
                    )

            for k in self.ks:
                if not vocab_k[k]:
                    raise ValueError(f"Input vocabulary doesn't contain any {k}mer.")

            return vocab_k
        return {k: None for k in self.ks}


class ToTensor:
    """Class to convert input to torch tensor"""

    def __init__(self, dtype: torch.dtype | None = None):
        """Initialize ToTensor

        Args:
            dtype (torch.dtype | None, optional): Desired dtype for the output. Defaults to None, meaning
            the dtype will be inferred from the data directly.
        """
        self.dtype = dtype

    def __call__(self, data: list | np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data, dtype=self.dtype)


class Compose:
    """Class to composes multiple Callable together to conduct transformation on the data
    sequentially."""

    def __init__(self, transforms: list[Callable]):
        """Initialize Compose

        Args:
            transforms (list[Callable]): A list of callable that define the transformation
        """
        for i, transform in enumerate(transforms):
            assert callable(transform), f"The {i}th transform {transform} is not callable."
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        """Apply the composed transforms to the input data sequentially

        Args:
            data (_type_): _description_

        Returns:
            Any: Transformed data after applying all the transforms.
        """
        for transform in self.transforms:
            data = transform(data)
        return data

    def __repr__(self) -> str:
        """Return a string representation of the Compose object."""
        transform_strings = [f"{transform.__name__}" for transform in self.transforms]
        return f"{self.__class__.__name__}({', '.join(transform_strings)})"
