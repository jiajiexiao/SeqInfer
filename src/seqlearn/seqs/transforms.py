from typing import Iterable, Mapping

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class GenerateKmerFreq:
    """Class to generate kmer frequency"""

    def __init__(
        self,
        k: int,
        vocabulary: Mapping | Iterable | None = None,
    ) -> None:
        """Constructor of GenerateKmerFreq

        Args:
            k (int): length of kmer.
            vocabulary (Mapping | Iterable, optional):
                vocabulary for kmers of interest. Either a Mapping (e.g., a
                dict) where keys are terms and values are indices in the
                feature matrix, or an iterable over terms. If not given, a
                vocabulary is determined from the inputs. Indices in the mapping
                should not be repeated and should not have any gap between 0 and
                the largest index.
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

    def __call__(self, seq: str) -> np.ndarray:
        # no fit happens when vocabulary is not None
        output = self.vectorizer.fit_transform([seq]).toarray()
        return output

    @property
    def kmer_names(self) -> np.ndarray:
        """kmer name for corresponding columns"""
        return self.vectorizer.get_feature_names_out()

    @property
    def kmer_index_mapping(self) -> dict[str, int]:
        """A mapping of kmers to feature indices"""
        return self.vectorizer.vocabulary_


class GenerateMultiKmerFreq:
    """Class to generate frequency of kmers with multiple values of k"""

    def __init__(
        self,
        ks: list[int],
        vocabulary: Mapping | Iterable | None = None,
    ):
        self.ks = ks
        self.vocabulary_k = self._get_vocabulary_k(vocabulary)
        self.kmer_freq_generators = [
            GenerateKmerFreq(k, vocabulary=self.vocabulary_k[k]) for k in self.ks
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

    def _get_vocabulary_k(
        self, vocabulary: Mapping | Iterable | None
    ) -> dict[int, Mapping | Iterable | None]:
        """helper method to parse input vocabulary for each possible value of k

        Args:
            vocabulary (Mapping | Iterable | None):
                vocabulary for kmers of interest. Check GenerateKmerFreq and
                sklearn's CountVectorizer to see more details about this argument.

        Raises:
            ValueError: A kmer in the vocabulary has a length outside the range of ks.
            ValueError: A value of k in ks is not found in the user-specified vocabulary.

        Returns:
            dict[int, Mapping | Iterable | None]: a dictionary contains sub-vocabulary per k.
        """
        if vocabulary:
            vocabulary_k = {k: [] for k in self.ks}
            for kmer in vocabulary:
                k = len(kmer)
                if k in self.ks:
                    vocabulary_k[k].append(kmer)
                else:
                    raise ValueError(
                        f"Input vocabulary contains {k}mer outside input range {self.ks}."
                    )

            for k in self.ks:
                if not vocabulary_k[k]:
                    raise ValueError(f"Input vocabulary doesn't contain any {k}mer.")

            return vocabulary_k
        return {k: None for k in self.ks}
